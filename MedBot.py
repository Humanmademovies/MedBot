from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, torch, pathlib, PyPDF2
from PIL import Image
import threading

def fallback_attention():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

class MedBot:
    """Assistant médical multimodal avec RAG (texte + PDF)."""

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        system_prompt_path: str = "system_prompt.md"
    ):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda",
        )
        self.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(dim)
        self.corpus: list[str] = []

    def add_docs(self, texts: list[str]):
        vecs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(vecs)
        self.corpus.extend(texts)

    def add_pdf(self, path: str | pathlib.Path, chunk: int = 512, overlap: int = 64):
        words = " ".join(p.extract_text() or "" for p in PyPDF2.PdfReader(path).pages).split()
        docs = [
            " ".join(words[i : i + chunk])
            for i in range(0, len(words), chunk - overlap)
            if words[i : i + chunk]
        ]
        self.add_docs(docs)

    def _retrieve(self, query: str, k: int) -> list[str]:
        if not self.corpus:
            return []
        v = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        _, I = self.index.search(v, min(k, len(self.corpus)))
        return [self.corpus[i] for i in I[0] if i >= 0]

    def chat(
        self,
        user_text: str,
        image: str | pathlib.Path | Image.Image | None = None,
        rag_k: int = 3,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        ctx = "\n".join(self._retrieve(user_text, rag_k)) if rag_k else ""
        txt = (f"Contexte:\n{ctx}\n\n" if ctx else "") + user_text

        content = [{"type": "text", "text": txt}]
        img_for_llm = None
        img_path = None
        if image is not None:
            if isinstance(image, (str, pathlib.Path)):
                img_path = str(image)
                img_for_llm = Image.open(image)
            else:
                img_path = "<uploaded image>"
                img_for_llm = image
            content.append({"type": "image", "image": img_path})

        self.messages.append({"role": "user", "content": content})

        # Corrige uniquement si le dernier message est du même rôle que l'avant-dernier
        if len(self.messages) >= 2:
            last_role = self.messages[-1]['role']
            prev_role = self.messages[-2]['role']
            if last_role == prev_role:
                missing_role = 'assistant' if last_role == 'user' else 'user'
                print(f"⚠️ Correction d'alternance : ajout d'un message factice ({missing_role})")
                self.messages.append({"role": missing_role, "content": [{"type": "text", "text": "..."}]})

        # PATCH : remplace le chemin par l'objet image dans le dernier message AVANT génération
        if img_for_llm is not None:
            self.messages[-1]["content"] = [
                part if part["type"] != "image" else {"type": "image", "image": img_for_llm}
                for part in self.messages[-1]["content"]
            ]

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        prompt_len = inputs["input_ids"].shape[-1]
        try:
            with torch.inference_mode():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                )[0][prompt_len:]
        except RuntimeError as e:
            if "p.attn_bias_ptr" in str(e):
                print("⚠️ Erreur d'alignement détectée. Bascule sur l'implémentation mathématique...")
                fallback_attention()
                with torch.inference_mode():
                    ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                    )[0][prompt_len:]
            else:
                raise

        reply = self.processor.decode(ids, skip_special_tokens=True).strip()

        # PATCH : remet le chemin dans le dernier message pour que la DB soit sérialisable
        if img_for_llm is not None:
            self.messages[-1]["content"] = [
                part if part["type"] != "image" else {"type": "image", "image": img_path}
                for part in self.messages[-1]["content"]
            ]

        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        return reply



    def stream_chat(
        self,
        user_text: str,
        image: str | pathlib.Path | None = None,
        rag_k: int = 3,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ):
        ctx = "\n".join(self._retrieve(user_text, rag_k)) if rag_k else ""
        txt = (f"Contexte:\n{ctx}\n\n" if ctx else "") + user_text

        content = [{"type": "text", "text": txt}]
        if image is not None:
            if isinstance(image, (str, pathlib.Path)):
                image = Image.open(image)
            content.append({"type": "image", "image": image})

        self.messages.append({"role": "user", "content": content})
        # Correction d’alternance sur self.messages
        if len(self.messages) >= 2:
            last_role = self.messages[-1]['role']
            prev_role = self.messages[-2]['role']
            if last_role == prev_role:
                missing_role = 'assistant' if last_role == 'user' else 'user'
                self.messages.append({"role": missing_role, "content": [{"type": "text", "text": "..."}]})

        inputs = self.processor.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.model.dtype)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        t = threading.Thread(
            target=torch_inference_generate,
            args=(self.model, inputs, streamer, max_new_tokens, temperature, top_p, repetition_penalty)
        )
        t.start()

        prev = ""
        for chunk in streamer:
            if not chunk.startswith(" ") and prev and not prev.endswith(" "):
                chunk = " " + chunk
            prev += chunk
            yield chunk

        t.join()

    def set_history(self, history):
        """Synchronise l'historique avec la DB (messages au bon format)."""
        self.messages = history.copy()


def torch_inference_generate(model, inputs, streamer, max_new_tokens, temperature, top_p, repetition_penalty):
    with torch.inference_mode():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
        )
