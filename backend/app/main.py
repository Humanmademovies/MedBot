from fastapi import FastAPI

app = FastAPI(title="MedGemma Prototype")

@app.get("/health")
async def health() -> dict[str, str]:
    """Point de contrôle basique de vitalité."""
    return {"status": "ok"}
