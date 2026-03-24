import asyncio
import os
import re
from typing import Optional

import httpx
from dotenv import load_dotenv

from ..core.base import BasePlatformCollector
from ..core.models import ModelDetail, ModelSummary, PlatformType
from ..core.registry import PlatformRegistry

load_dotenv()

_BASE_URL = "https://civitai.com"


def _strip_html(text: str) -> str:
    """HTML 태그를 제거하고 일반 텍스트를 반환합니다."""
    clean = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", clean).strip()


@PlatformRegistry.register
class CivitaiCollector(BasePlatformCollector):
    platform_name = "civitai"
    platform_display_name = "Civitai"

    def __init__(self) -> None:
        token = os.getenv("CIVITAI_API_TOKEN")
        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            headers=headers,
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(3):
            resp = await self._client.get(path, params=params)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5))
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Rate limit 초과 - 잠시 후 다시 시도하세요.")

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            await self._get("/api/v1/models", {"limit": 1})
            return True
        except Exception:
            return False

    async def list_models(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> list[ModelSummary]:
        params: dict = {
            "limit": limit,
            "sort": kwargs.get("sort", "Most Downloaded"),
            "period": kwargs.get("period", "AllTime"),
        }
        if query:
            params["query"] = query
        if kwargs.get("types"):
            params["types"] = kwargs["types"]

        data = await self._get("/api/v1/models", params)
        items = data.get("items") or []

        results: list[ModelSummary] = []
        for item in items:
            creator = item.get("creator") or {}
            stats = item.get("stats") or {}
            desc = item.get("description") or ""
            results.append(
                ModelSummary(
                    platform=PlatformType.CIVITAI,
                    model_id=str(item.get("id", "")),
                    name=item.get("name", ""),
                    author=creator.get("username"),
                    description=_strip_html(desc)[:300] if desc else None,
                    tags=item.get("tags") or [],
                    downloads=stats.get("downloadCount"),
                    likes=stats.get("favoriteCount"),
                    nsfw=item.get("nsfw"),
                )
            )
        return results

    async def get_model_detail(self, model_id: str) -> ModelDetail:
        item = await self._get(f"/api/v1/models/{model_id}")

        creator = item.get("creator") or {}
        stats = item.get("stats") or {}
        desc = item.get("description") or ""

        # 최신 버전 기준
        versions = item.get("modelVersions") or []
        latest = versions[0] if versions else {}
        files = latest.get("files") or []
        first_file = files[0] if files else {}
        file_meta = first_file.get("metadata") or {}

        trained_words = latest.get("trainedWords") or []

        return ModelDetail(
            platform=PlatformType.CIVITAI,
            model_id=str(item.get("id", model_id)),
            name=item.get("name", ""),
            author=creator.get("username"),
            description=_strip_html(desc) if desc else None,
            tags=item.get("tags") or [],
            downloads=stats.get("downloadCount"),
            likes=stats.get("favoriteCount"),
            nsfw=item.get("nsfw"),
            # detail
            pipeline_tag=item.get("type"),
            base_model=latest.get("baseModel"),
            format=file_meta.get("format"),
            quantization=file_meta.get("size"),
            training_datasets=trained_words,
            raw_data=item,
        )
