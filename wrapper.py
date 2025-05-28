"""wrapper5.py – Stand‑alone SVF wrapper (v10)
===========================================================
État intégral, indentation vérifiée au linter.
• Advantage + baseline EMA.
• Micro‑batch dynamique (token budget).
• Cast float32 pour softmax/température.
• Pas de placeholder KL (facile à ré‑ajouter).
• Projection ∇W→∇α avec max_mult.
"""
from __future__ import annotations
import copy
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from base3 import Policy  
import os
from Tracker import ProgressTracker     
# -----------------------------------------------------------------------------
@torch.no_grad()
def iter_token_batches(prompt_l: list[int], resp_l: list[int], max_tokens: int = 2048):
    """Regroupe les indices pour que Σ tokens (prompt+réponse) ≤ max_tokens."""
    batch, tot = [], 0
    for i, (pl, rl) in enumerate(zip(prompt_l, resp_l)):
        need = pl + rl
        if batch and tot + need > max_tokens:
            yield batch
            batch, tot = [], 0
        batch.append(i)
        tot += need
    if batch:
        yield batch

@torch.no_grad()
def _filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0):
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = torch.where(logits < kth, logits.new_full((), -float('inf')), logits)
    if 0.0 < top_p < 1.0:
        probs = logits.softmax(-1)
        sorted_p, idx = torch.sort(probs, descending=True)
        cumsum = sorted_p.cumsum(-1)
        mask = cumsum > top_p; mask[..., 0] = 0
        logits.scatter_(dim=-1, index=idx, src=torch.where(mask, logits.new_full((), -float('inf')), logits))
    return logits

def is_linear(key: str) -> bool:
    key_l = key.lower()
    if not key.endswith(".weight"):
        return False
    if any(bad in key_l for bad in ("embed", "embedding", "embed_tokens")):
        return False
    if "norm" in key_l or "layernorm" in key_l or ".ln_" in key_l:
        return False
    return True



# -----------------------------------------------------------------------------
class HFOpenAIWrapper:
    """LLM wrapper + SVD-REINFORCE (mémoire friendly)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_ref_model: bool = False,
        *,
        lr: float = 2e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,   # ← NEW
        tracker_kwargs: Optional[dict] = None,
    ):
        # ----------- Chargement backbone ------------------------------------
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # ----------- Attributs internes -------------------------------------
        self.decomposed_params: dict[str, dict[str, torch.Tensor]] = {}
        self.policy: Policy | None = None
        self.running_reward = torch.tensor(0.0, device=self.device)

        if load_ref_model:
            self.ref_model = copy.deepcopy(self.model).eval().requires_grad_(False)
            self.ref_model.to(self.device)

        # ----------- SVD des poids linéaires --------------------------------
        self.expert_name = "expert"
        self.svd_path = self._get_svd_path()
        if os.path.exists(self.svd_path):
            print(f"[SVD] Loading decomposed params from {self.svd_path}")
            self.decomposed_params = torch.load(self.svd_path, map_location="cpu")
        else:
            print("[SVD] Decomposing linear weights and saving…")
            self.svd_decompose_params_auto()
            torch.save(self.decomposed_params, self.svd_path)

        # ----------- Politique α (trainable) --------------------------------
        self.init_svd_policy()                     # crée self.policy

        # ----------- Optimiseur ---------------------------------------------
        # Si l’utilisateur n’en fournit pas, on crée un Adam par défaut.
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(self.policy.trainable_params, lr=lr)
        )

        # ----------- Tracker de progression ---------------------------------
        self.tracker = ProgressTracker(
            wrapper=self,
            optimizer=self.optimizer,
            **(tracker_kwargs or {}),
        )


    # ---------------- SVD & policy ----------------
    def svd_decompose_params(self, keys: List[str]):
        for n, p in self.model.named_parameters():
            if n in keys:
                U, S, Vh = torch.linalg.svd(p.data.float(), full_matrices=False)
                self.decomposed_params[n] = {'U': U.cpu(), 'S': S.cpu(), 'Vh': Vh.cpu()}
                p.requires_grad = True
            else:
                p.requires_grad = False

    def init_svd_policy(self, init_val: float = 0.0, max_mult: float = 2.0):
        base_S = {k: v['S'] for k, v in self.decomposed_params.items()}
        self.policy = Policy(base_S, gpu=self.device, init_val=init_val, max_mult=max_mult)

    # ---------------- mask utilities ----------------
    @torch.no_grad()
    def _compose_masked_tensor(self, name: str):
        svd = self.decomposed_params[name]
        U, S, Vh = (svd['U'].to(self.device), svd['S'].to(self.device), svd['Vh'].to(self.device))
        alpha = self.policy.learnable_params[name]
        mask = self.policy.get_mask(alpha)
        Wp = U @ torch.diag(S * mask) @ Vh
        scale = S.sum() / ((S * mask).sum() + 1e-8)
        return (Wp * scale).to(torch.bfloat16)

    @torch.no_grad()
    def apply_svd_masks(self):
        for full in self.decomposed_params:
            new_W = self._compose_masked_tensor(full)
            mod = self.model
            for part in full.split('.')[:-1]:
                mod = getattr(mod, part)
            getattr(mod, full.split('.')[-1]).data.copy_(new_W)

    # ---------------- generation ----------------
    def _generate_onepass(self, prompts: List[str], *, max_new_tokens=128, temperature=0.7, apply_mask=True):
        tok = self.tokenizer
        tok.padding_side = 'left'
        tok.pad_token_id = tok.pad_token_id or tok.eos_token_id
        if apply_mask and self.policy:
            self.apply_svd_masks()
        rendered = [tok.apply_chat_template([
            {'role': 'system', 'content': p[0]},
            {'role': 'user',   'content': p[1]}], add_generation_prompt=True, tokenize=False) for p in prompts]
        enc = tok(rendered, padding=True, return_tensors='pt').to(self.device)
        prompt_lens = enc.attention_mask.sum(1).tolist()
        ids, attn = enc.input_ids.clone(), enc.attention_mask.clone()
        done = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)
        logbuf: list[list[torch.Tensor]] = [[] for _ in prompts]
        past = None
        with torch.no_grad(), autocast('cuda', torch.bfloat16):
            for _ in range(max_new_tokens):
                inp = ids if past is None else ids[:, -1:]
                out = self.model(inp, attention_mask=attn, past_key_values=past, use_cache=True, return_dict=True)
                past = out.past_key_values
                logits = (out.logits[:, -1] / temperature).float()
                logits = _filter_logits(logits)
                probs = logits.softmax(-1)
                tok_id = torch.multinomial(probs, 1).squeeze(-1)
                logp = probs.gather(-1, tok_id[:, None]).log().squeeze(-1)
                tok_id = torch.where(done, tok.pad_token_id, tok_id)
                logp = torch.where(done, logp.new_zeros(()), logp)
                ids = torch.cat([ids, tok_id[:, None]], 1)
                attn = torch.cat([attn, (~done)[:, None].long()], 1)
                for b in range(len(prompts)):
                    if not done[b]:
                        logbuf[b].append(logp[b])
                done |= tok_id.eq(tok.eos_token_id)
                if done.all():
                    break
        log_seqs = [torch.stack(seq) if seq else torch.empty(0, device=self.device) for seq in logbuf]
        return ids, log_seqs, prompt_lens

        # ---------------------------------------------------------------------
    def batch_generate_with_logprobs(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 128,
        temperature: float | None = 0.7,          # peut être None désormais
        apply_mask: bool = True,
        gen_batch_size: int = 16,
        tracker: "ProgressTracker | None" = None, # new (optionnel)
    ):
        """
        Génère des réponses + log-p token-wise pour une liste de prompts.

        - Si `temperature` vaut None, on essaie de récupérer
          tracker.temperature, puis self.tracker.temperature,
          sinon on retombe sur 0.7.
        """
        # -------- température par défaut ---------------------------------
        if temperature is None:
            if tracker is not None and hasattr(tracker, "temperature"):
                temperature = tracker.temperature
            elif getattr(self, "tracker", None) is not None:
                temperature = self.tracker.temperature
            else:
                temperature = 0.7

        responses, logps = [], []
        for s in range(0, len(prompts), gen_batch_size):
            ids, lp, pl = self._generate_onepass(
                prompts[s : s + gen_batch_size],
                max_new_tokens=max_tokens,
                temperature=temperature,
                apply_mask=apply_mask,
            )
            responses += [
                self.tokenizer.decode(row[pl_i:], skip_special_tokens=True)
                for row, pl_i in zip(ids, pl)
            ]
            logps += lp
            torch.cuda.empty_cache()
        return responses, logps


    # ---------------- REINFORCE ----------------
        # ---------------------------------------------------------------------
    def reinforce_batch_step_weighted(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        optimizer: Optional[torch.optim.Optimizer] = None,  # ← peut être omis
        *,
        max_tokens_tf: int = 2048,
        kl_ref_coeff: float = 0.0,
        ref_model: torch.nn.Module | None = None,
    ) -> float:
        """
        REINFORCE + baseline EMA + (optionnel) KL(p_ref || p_theta).

        Args
        ----
        prompts / responses : listes alignées.
        rewards             : scalaires positifs ou négatifs.
        optimizer           : instance torch.optim.
        max_tokens_tf       : budget de tokens pour le backward → grad ∇θ.
        kl_ref_coeff        : pondération λ du terme KL. 0.0 => pas de régul.
        ref_model           : modèle de référence (non masqué, eval mode).
                              Si None, on utilise self.ref_model.
        REINFORCE + baseline EMA + (optionnel) KL(p_ref || p_theta).

        Si `optimizer` n'est pas fourni, on utilise `self.optimizer`.

        Returns
        -------
        float : perte moyenne (policy-gradient + KL) sur tout le batch.
        """
        if not rewards:
            return 0.0

        # ----------- optimiseur par défaut --------------------------------
        optimizer = optimizer or getattr(self, "optimizer", None)
        if optimizer is None:
            raise RuntimeError("Aucun optimiseur fourni et self.optimizer absent.")

        if ref_model is None:
            ref_model = getattr(self, "ref_model", None)
        if kl_ref_coeff > 0.0 and ref_model is None:
            raise ValueError(
                "kl_ref_coeff > 0 mais aucun ref_model fourni / stocké."
            )

        # ----------- advantage : reward – baseline EMA -------------------
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        r = (r - r.mean()) / (r.std() + 1e-8)
        adv = r

        # Comptage de tokens pour découper en micro-lots « safe »
        prompt_tok = [len(self.tokenizer.tokenize(p)) for p in prompts]
        resp_tok   = [len(self.tokenizer.tokenize(a)) for a in responses]

        optimizer.zero_grad()
        total_loss = 0.0

        for idxs in iter_token_batches(prompt_tok, resp_tok, max_tokens_tf):
            tok = self.tokenizer
            seqs = [
                tok.apply_chat_template(
                    [
                        {"role": "system",    "content": "You are a helpful assistant."},
                        {"role": "user",      "content": prompts[i]},
                        {"role": "assistant", "content": responses[i]},
                    ],
                    tokenize=False,
                )
                for i in idxs
            ]
            enc = tok(seqs, padding=True, return_tensors="pt").to(self.device)

            # ---------------- forward modèle courant ---------------------
            with autocast("cuda", torch.bfloat16):
                logits = self.model(**enc, use_cache=False).logits[:, :-1].float()

            tgt     = enc.input_ids[:, 1:]                       # next-token targets
            logtok  = F.log_softmax(logits, dim=-1)              # log p_θ
            tok_logp = logtok.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            seq_logp = tok_logp.sum(1) / tgt.size(1)             # moyenne par séquence

            pg_loss = (-seq_logp * adv[idxs]).mean()             # policy-gradient

            # ---------------- KL régularisation -------------------------
            kl_loss = torch.tensor(0.0, device=self.device)
            if kl_ref_coeff > 0.0:
                with torch.no_grad():
                    ref_logits = ref_model(**enc, use_cache=False).logits[:, :-1].float()
                    ref_logtok = F.log_softmax(ref_logits, dim=-1)  # log p_ref

                kl_loss = F.kl_div(
                    input=logtok,
                    target=ref_logtok,
                    log_target=True,
                    reduction="batchmean",
                )

            loss = pg_loss + kl_ref_coeff * kl_loss
            loss.backward()
            total_loss += loss.item() * len(idxs)

            torch.cuda.empty_cache()

        # ----------- projeter ∇W → ∇α puis update ------------------------
        self.project_grads_to_alpha()
        torch.nn.utils.clip_grad_norm_(self.policy.trainable_params, 1e-3)
        optimizer.step()

        return total_loss / len(prompts)


    # ---------------- projection ----------------
    def project_grads_to_alpha(self):
        for name, svd in self.decomposed_params.items():
            if name not in self.policy.learnable_params:   # α existe ?
                continue

            U = svd['U'].to(self.device)
            S = svd['S'].to(self.device)
            Vh = svd['Vh'].to(self.device)
            # récupère le grad du poids reconstruit
            mod = self.model
            for part in name.split('.')[:-1]:
                mod = getattr(mod, part)
            dLdW = getattr(mod, name.split('.')[-1]).grad
            if dLdW is None:
                continue
            dLdS = torch.diag(U.T @ dLdW.float() @ Vh.T)
            alpha = self.policy.learnable_params[name]
            mask  = self.policy.get_mask(alpha)
            dLdA = dLdS * S * mask * (1 - mask / self.policy.max_mult)

            if alpha.grad is None:
                alpha.grad = dLdA.to(alpha.dtype)
            else:
                alpha.grad.copy_(dLdA.to(alpha.dtype))

    def _get_expert_dir(self) -> str:
        expert_dir = os.path.join(os.getcwd(), "experts")
        os.makedirs(expert_dir, exist_ok=True)
        return expert_dir

    def _get_expert_filename(self, expert_name: str) -> str:
        model_short_name = self.model.config._name_or_path.split("/")[-1]
        filename = f"{model_short_name}_{expert_name}.pt"
        return os.path.join(self._get_expert_dir(), filename)

    def save_expert(self, expert_name: str = "expert", *,
                lightweight: bool = True):
        """
        Sauvegarde l'expert.
        – lightweight=True  ➜  ne stocke que les α (~ |S| floats)
        – lightweight=False ➜  ancien comportement (full dump)
        """
        path = self._get_expert_filename(expert_name)

        obj = {'policy_state': self.policy.state_dict()}
        if not lightweight:
            obj['decomposed_params'] = self.decomposed_params  # rétro-compat

        torch.save(obj, path)
        print(f"[Expert] Saved to {path} ({'lite' if lightweight else 'full'})")

    def load_expert(self, expert_name: str):
        path = self._get_expert_filename(expert_name)
        ckpt = torch.load(path, map_location=self.device)

        # SVD déjà présent grâce à __init__ ; sinon on l'initialise
        if not self.decomposed_params:
            self.decomposed_params = torch.load(self.svd_path, map_location="cpu")

        # certains anciens checkpoints contiennent encore decomposed_params
        if 'decomposed_params' in ckpt:
            self.decomposed_params = ckpt['decomposed_params']

        self.init_svd_policy()                 # crée un Policy vide
        self.policy.load_state_dict(ckpt['policy_state'])
        self.apply_svd_masks()
        print(f"[Expert] '{expert_name}' loaded.")


    def get_expert_list(self):
        expert_dir = self._get_expert_dir()
        model_short_name = self.model.config._name_or_path.split("/")[-1]
        experts = []
        for file in os.listdir(expert_dir):
            if file.startswith(model_short_name + "_") and file.endswith(".pt"):
                expert = file[len(model_short_name) + 1:-3]  # retirer préfixe et .pt
                experts.append(expert)
        return experts
    
    def save_current_state(self):
        cpu_decomposed = {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in self.decomposed_params.items()}
        cpu_policy = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        return {
            'decomposed_params': cpu_decomposed,
            'policy_state': cpu_policy,
        }

    def load_state_from(self, state):
        self.decomposed_params = state['decomposed_params']
        self.init_svd_policy()
        self.policy.load_state_dict(state['policy_state'])
        self.apply_svd_masks()
        print("[State] Expert state restored from memory.")

    def svd_decompose_params_auto(self):
        for name, param in self.model.named_parameters():
            if is_linear(name):
                U, S, Vh = torch.linalg.svd(param.data.float(), full_matrices=False)
                self.decomposed_params[name] = {
                    'U': U.cpu(), 'S': S.cpu(), 'Vh': Vh.cpu()
                }
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _get_svd_path(self) -> str:
        model_name = self.model.config._name_or_path.split("/")[-1]
        folder = os.path.join("models", model_name, "decomposed")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{model_name}_decomposed.pt")