import os
from typing import Optional

import httpx
from dotenv import load_dotenv

from ..core.base import BasePlatformCollector
from ..core.models import ModelDetail, ModelSummary, PlatformType
from ..core.registry import PlatformRegistry

load_dotenv()


@PlatformRegistry.register
class OllamaCollector(BasePlatformCollector):
    platform_name = "ollama"
    platform_display_name = "Ollama"

    def __init__(self) -> None:
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False
        except Exception:
            return False

    async def list_models(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> list[ModelSummary]:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError:
            raise ConnectionError(
                "Ollama 서버에 연결할 수 없습니다. `ollama serve` 명령으로 서버를 먼저 실행해 주세요."
            )

        data = resp.json()
        models = data.get("models") or []

        results: list[ModelSummary] = []
        for item in models:
            name = item.get("name") or item.get("model", "")
            model_id = item.get("model") or name

            # 클라이언트 측 필터링
            if query and query.lower() not in name.lower():
                continue

            details = item.get("details") or {}
            results.append(
                ModelSummary(
                    platform=PlatformType.OLLAMA,
                    model_id=model_id,
                    name=name,
                    tags=[details.get("family", "")] if details.get("family") else [],
                    last_modified=item.get("modified_at"),
                )
            )
            if len(results) >= limit:
                break

        return results

    async def get_model_detail(self, model_id: str) -> ModelDetail:
        try:
            resp = await self._client.post(
                "/api/show",
                json={"model": model_id, "verbose": True},
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise ConnectionError(
                "Ollama 서버에 연결할 수 없습니다. `ollama serve` 명령으로 서버를 먼저 실행해 주세요."
            )

        item = resp.json()
        details = item.get("details") or {}
        model_info = item.get("model_info") or {}
        capabilities = item.get("capabilities") or []

        tags: list[str] = []
        for fam in details.get("families") or []:
            tags.append(fam)
        for cap in capabilities:
            tags.append(cap)

        return ModelDetail(
            platform=PlatformType.OLLAMA,
            model_id=model_id,
            name=model_id,
            tags=tags,
            parameter_count=details.get("parameter_size"),
            architecture=details.get("family"),
            quantization=details.get("quantization_level"),
            format=details.get("format", "gguf"),
            license=item.get("license"),
            model_card=item.get("modelfile"),
            raw_data={
                "template": item.get("template"),
                "parameters": item.get("parameters"),
                "model_info": model_info,
            },
        )
