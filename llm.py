# llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from typing import Iterator, Optional

class TorchChatEngine:
    """
    Moteur d'inférence PyTorch minimal (HF Transformers).
    - charge un modèle instruct LLM
    - génère en streaming (yield token par token)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",              # "cuda" | "cpu"
        dtype: torch.dtype = torch.float16 # bfloat16 ou float16 pour CUDA, float32 pour CPU
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

    def stream_chat(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=[self.tokenizer.eos_token_id]
            + [self.tokenizer.convert_tokens_to_ids(t) for t in stop or []],
        )
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


if __name__ == "__main__":
    engine = TorchChatEngine(device="cuda" if torch.cuda.is_available() else "cpu")
    prompt = "### Instruction:\nExplique brièvement la photosynthèse.\n\n### Réponse:"
    print("".join(engine.stream_chat(prompt)))

