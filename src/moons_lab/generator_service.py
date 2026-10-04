from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .generator import GeneratorSettings, LiveGeneratorController


class ConfigPatch(BaseModel):
    mode: str | None = None
    n_samples: int | None = None
    seed: int | None = None
    noise: float | None = None
    shift_x: float | None = None
    shift_y: float | None = None
    rotation_deg: float | None = None
    scale: float | None = None
    class_1_ratio: float | None = None
    label_flip_rate: float | None = None
    random_range: float | None = None


class GenerateRequest(BaseModel):
    n_samples: int | None = None
    seed: int | None = None


def create_generator_app(controller: LiveGeneratorController | None = None) -> FastAPI:
    controller = controller or LiveGeneratorController()
    app = FastAPI(title="make_moons live generator")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def get_config() -> dict:
        return controller.get_settings().model_dump()

    @app.patch("/config")
    def patch_config(patch: ConfigPatch) -> dict:
        values = {k: v for k, v in patch.model_dump().items() if v is not None}
        return controller.update(values).model_dump()

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        frame = controller.generate(n_samples=req.n_samples, seed=req.seed)
        return {"rows": len(frame), "records": frame.to_dict(orient="records")}

    return app
