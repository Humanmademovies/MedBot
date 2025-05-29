from mm_engine import Engine
from PIL import Image
import io
import base64

def _decode_base64_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes))

class MedBot:
    def __init__(self, model_id="google/medgemma-4b-it", system_prompt_path="system_prompt.md"):
        self.engine = Engine(model_id)
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        self.messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]

    def set_history(self, messages):
        self.messages = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                self.messages.append(msg)

    def _ensure_alternance(self):
        filtered = [m for m in self.messages if m["role"] in ["user", "assistant"]]
        for i in range(len(filtered) - 1):
            if filtered[i]["role"] == filtered[i + 1]["role"]:
                missing_role = "assistant" if filtered[i]["role"] == "user" else "user"
                self.messages.append({"role": missing_role, "content": [{"type": "text", "text": "..."}]})
                break


    def chat(self, messages, image=None, stream=False):
        img_for_engine = None
        if image is not None:
            img_for_engine = _decode_base64_image(image)

        result = self.engine.chat(messages=messages, images=[img_for_engine] if img_for_engine else None, stream=stream)

        if stream:
            accumulated = ""
            for chunk in result:
                accumulated += chunk
                yield chunk
            return accumulated
        else:
            return result
