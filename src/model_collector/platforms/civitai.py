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


def _scan_status(result: str | None) -> str:
    """Civitai 스캔 결과 문자열을 표준 상태로 변환합니다."""
    if result in ("Success",):
        return "safe"
    if result in ("Pending", None, ""):
        return "unscanned"
    return "dangerous"


def _overall_status(statuses: list[str]) -> str:
    if any(s == "dangerous" for s in statuses):
        return "dangerous"
    if any(s == "unscanned" for s in statuses):
        return "unscanned"
    return "safe"


def _build_security(files: list[dict]) -> tuple[list[dict], dict]:
    """파일 목록에서 security_file_details와 security_status를 구성합니다."""
    details: list[dict] = []
    for f in files:
        pkl_result = f.get("pickleScanResult")
        pkl_msg = f.get("pickleScanMessage") or ""
        av_result = f.get("virusScanResult")
        av_msg = f.get("virusScanMessage") or ""

        pkl_status = _scan_status(pkl_result)
        av_status = _scan_status(av_result)
        overall = _overall_status([pkl_status, av_status])

        entry: dict = {
            "path": f.get("name", ""),
            "overall_status": overall,
            "pickle_scan": {"status": pkl_status, "message": pkl_msg},
            "virustotal": {"status": av_status, "message": av_msg},
        }
        details.append(entry)

    files_with_issues = [d["path"] for d in details if d["overall_status"] != "safe"]
    status = {
        "scansDone": True,
        "filesWithIssues": files_with_issues,
    }
    return details, status


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

            # created_at: 첫 번째 버전의 publishedAt 사용
            versions = item.get("modelVersions") or []
            created_at = None
            if versions:
                raw_date = versions[0].get("publishedAt")
                if raw_date:
                    try:
                        from datetime import datetime, timezone
                        created_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                    except Exception:
                        pass

            results.append(
                ModelSummary(
                    platform=PlatformType.CIVITAI,
                    model_id=str(item.get("id", "")),
                    name=item.get("name", ""),
                    author=creator.get("username"),
                    description=_strip_html(desc)[:300] if desc else None,
                    tags=item.get("tags") or [],
                    downloads=stats.get("downloadCount"),
                    likes=stats.get("thumbsUpCount"),
                    trending_score=float(stats.get("tippedAmountCount", 0) or 0),
                    nsfw=item.get("nsfw"),
                    created_at=created_at,
                )
            )
        return results

    async def get_model_detail(self, model_id: str) -> ModelDetail:
        item = await self._get(f"/api/v1/models/{model_id}")

        creator = item.get("creator") or {}
        stats = item.get("stats") or {}
        desc = item.get("description") or ""

        # 버전 목록
        versions = item.get("modelVersions") or []
        latest = versions[0] if versions else {}
        files = latest.get("files") or []

        # 기본 파일(primary=True 우선, 없으면 첫 번째)
        primary_file = next((f for f in files if f.get("primary")), files[0] if files else {})
        file_meta = primary_file.get("metadata") or {}

        trained_words = latest.get("trainedWords") or []

        # 보안 스캔 정보 (최신 버전 파일 기준)
        security_file_details, security_status = _build_security(files)

        # 아키텍처/하이퍼파라미터
        arch_hyperparams: dict = {}
        if file_meta.get("fp"):
            arch_hyperparams["FP 정밀도"] = file_meta["fp"]
        if file_meta.get("size"):
            arch_hyperparams["가중치 크기"] = file_meta["size"]
        if latest.get("baseModelType"):
            arch_hyperparams["베이스 모델 타입"] = latest["baseModelType"]
        arch_hyperparams["생성 지원"] = "예" if latest.get("supportsGeneration") else "아니오"
        if item.get("nsfwLevel") is not None:
            arch_hyperparams["NSFW 레벨"] = item["nsfwLevel"]

        # 저장 용량 (bytes)
        size_kb = primary_file.get("sizeKB")
        storage_size = int(size_kb * 1024) if size_kb else None

        # 파일 이름 목록
        model_files = [f.get("name", "") for f in files if f.get("name")]

        # 라이선스 문자열
        commercial = item.get("allowCommercialUse", "")
        # allowCommercialUse can be a JSON-like string "{Image,RentCivit}" or a plain string
        if isinstance(commercial, str):
            commercial = commercial.strip("{}")
        license_str = (
            f"상업적 사용: {commercial} / "
            f"파생 허용: {item.get('allowDerivatives', False)}"
        )

        # 추론 상태
        inference_status = "지원" if item.get("supportsGeneration") else "미지원"

        # 접근 제한 여부
        gated = item.get("availability") == "EarlyAccess"

        # 버전 요약 목록
        model_versions = [
            {
                "name": v.get("name", ""),
                "base_model": v.get("baseModel", ""),
                "downloads": (v.get("stats") or {}).get("downloadCount", 0),
                "published_at": v.get("publishedAt", ""),
                "file_count": len(v.get("files") or []),
            }
            for v in versions
        ]

        # created_at
        created_at = None
        raw_date = latest.get("publishedAt")
        if raw_date:
            try:
                from datetime import datetime
                created_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except Exception:
                pass

        return ModelDetail(
            platform=PlatformType.CIVITAI,
            model_id=str(item.get("id", model_id)),
            name=item.get("name", ""),
            author=creator.get("username"),
            description=_strip_html(desc) if desc else None,
            tags=item.get("tags") or [],
            downloads=stats.get("downloadCount"),
            likes=stats.get("thumbsUpCount"),
            trending_score=float(stats.get("tippedAmountCount", 0) or 0),
            nsfw=item.get("nsfw"),
            created_at=created_at,
            # detail
            pipeline_tag=item.get("type"),
            base_model=latest.get("baseModel"),
            format=file_meta.get("format"),
            quantization=file_meta.get("size"),
            training_datasets=trained_words,
            arch_hyperparams=arch_hyperparams if arch_hyperparams else None,
            model_card=_strip_html(desc) if desc else None,
            storage_size=storage_size,
            model_files=model_files,
            license=license_str,
            inference_status=inference_status,
            gated=gated,
            security_file_details=security_file_details,
            security_status=security_status,
            model_versions=model_versions,
            raw_data=item,
        )
