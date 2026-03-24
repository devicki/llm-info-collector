import asyncio
import os
import re
from typing import Optional

import httpx
from dotenv import load_dotenv

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from ..core.base import BasePlatformCollector
from ..core.models import ModelDetail, ModelSummary, PlatformType
from ..core.registry import PlatformRegistry

load_dotenv()

_BASE_URL = "https://huggingface.co"

_LIST_EXPAND = ["downloadsAllTime", "likes", "trendingScore", "inference"]
_DETAIL_EXPAND = [
    "cardData", "config", "safetensors", "tags",
    "downloadsAllTime", "likes", "trendingScore",
    "siblings", "spaces", "inference", "usedStorage",
    "transformersInfo",
]


@PlatformRegistry.register
class HuggingFaceCollector(BasePlatformCollector):
    platform_name = "huggingface"
    platform_display_name = "Hugging Face"

    def __init__(self) -> None:
        token = os.getenv("HF_API_TOKEN")
        headers = {"Accept": "application/json"}
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

    # 파일별 보안 상세 조회 대상 확장자
    _SECURITY_EXTENSIONS = {
        ".bin", ".pt", ".pth", ".ckpt", ".safetensors",
        ".gguf", ".msgpack", ".pkl", ".pickle", ".h5",
    }

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        for attempt in range(3):
            resp = await self._client.get(path, params=params)
            if resp.status_code == 429:
                wait = float(resp.headers.get("X-RateLimit-Reset", resp.headers.get("Retry-After", 5)))
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Rate limit 초과 - 잠시 후 다시 시도하세요.")

    async def _get_file_security_details(
        self, model_id: str, file_paths: list[str]
    ) -> list[dict]:
        """paths-info API로 파일별 상세 보안 스캔 결과를 가져옵니다."""
        if not file_paths:
            return []
        try:
            resp = await self._client.post(
                f"/api/models/{model_id}/paths-info/main",
                data={"paths": file_paths, "expand": True},
            )
            resp.raise_for_status()
            results = []
            for entry in resp.json():
                sec = entry.get("securityFileStatus")
                if not sec:
                    continue
                results.append({
                    "path": entry.get("path"),
                    "size": entry.get("size"),
                    "overall_status": sec.get("status"),
                    "protect_ai": sec.get("protectAiScan"),
                    "av_scan": sec.get("avScan"),
                    "pickle_scan": sec.get("pickleImportScan"),
                    "virustotal": sec.get("virusTotalScan"),
                    "jfrog": sec.get("jFrogScan"),
                })
            return results
        except Exception:
            return []

    async def _fetch_model_card(self, model_id: str) -> tuple[Optional[str], Optional[dict]]:
        """README.md(Model Card) 원문과 YAML frontmatter를 반환합니다."""
        try:
            resp = await self._client.get(
                f"/{model_id}/resolve/main/README.md",
                headers={"Accept": "text/plain"},
            )
            if resp.status_code != 200:
                return None, None
            text = resp.text

            # YAML frontmatter 파싱: --- ... --- 블록
            frontmatter: Optional[dict] = None
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
            if match and _YAML_AVAILABLE:
                try:
                    frontmatter = yaml.safe_load(match.group(1))
                except Exception:
                    pass
            elif match:
                # yaml 미설치 시 raw 텍스트를 dict 대신 저장
                frontmatter = {"_raw": match.group(1)}

            return text, frontmatter
        except Exception:
            return None, None

    @staticmethod
    def _parse_param_count(safetensors: dict) -> Optional[str]:
        total = safetensors.get("total")
        if not total:
            return None
        billions = total / 1e9
        return f"{billions:.1f}B" if billions >= 1 else f"{total / 1e6:.0f}M"

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        try:
            await self._get("/api/models", {"limit": 1})
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
            "pipeline_tag": kwargs.get("pipeline_tag", "text-generation"),
            "sort": kwargs.get("sort", "downloads"),
            "direction": -1,
            "limit": limit,
            "expand[]": _LIST_EXPAND,
        }
        if query:
            params["search"] = query

        data = await self._get("/api/models", params)
        results: list[ModelSummary] = []
        for item in data:
            results.append(
                ModelSummary(
                    platform=PlatformType.HUGGINGFACE,
                    model_id=item.get("id", ""),
                    name=item.get("modelId") or item.get("id", ""),
                    author=item.get("author"),
                    tags=item.get("tags") or [],
                    downloads=item.get("downloadsAllTime") or item.get("downloads"),
                    likes=item.get("likes"),
                    trending_score=item.get("trendingScore"),
                    last_modified=item.get("lastModified"),
                )
            )
        return results

    async def get_model_detail(self, model_id: str) -> ModelDetail:
        params = {
            "expand[]": _DETAIL_EXPAND,
            "securityStatus": "true",
        }
        item = await self._get(f"/api/models/{model_id}", params)

        card = item.get("cardData") or {}
        config = item.get("config") or {}
        safetensors = item.get("safetensors") or {}
        transformers_info = item.get("transformersInfo") or {}
        tokenizer_config = config.get("tokenizer_config") or {}
        siblings = item.get("siblings") or []
        spaces = item.get("spaces") or []

        languages = card.get("language") or card.get("languages") or []
        if isinstance(languages, str):
            languages = [languages]

        datasets = card.get("datasets") or card.get("dataset") or []
        if isinstance(datasets, str):
            datasets = [datasets]

        model_files = [s["rfilename"] for s in siblings if "rfilename" in s]

        # 보안 스캔 대상: 이슈 파일 우선, 없으면 주요 모델 파일 전체 (최대 20개)
        sec_repo = item.get("securityRepoStatus") or {}
        flagged = [f["path"] for f in (sec_repo.get("filesWithIssues") or [])]
        if flagged:
            scan_targets = flagged
        else:
            scan_targets = [
                f for f in model_files
                if any(f.endswith(ext) for ext in self._SECURITY_EXTENSIONS)
            ][:20]
        file_security = await self._get_file_security_details(model_id, scan_targets)
        model_card, card_frontmatter = await self._fetch_model_card(model_id)

        return ModelDetail(
            platform=PlatformType.HUGGINGFACE,
            model_id=item.get("id", model_id),
            name=item.get("modelId") or item.get("id", model_id),
            author=item.get("author"),
            description=card.get("model_description"),
            tags=item.get("tags") or [],
            downloads=item.get("downloadsAllTime") or item.get("downloads"),
            likes=item.get("likes"),
            trending_score=item.get("trendingScore"),
            last_modified=item.get("lastModified"),
            # 아키텍처
            parameter_count=self._parse_param_count(safetensors),
            architecture=config.get("model_type"),
            architectures=config.get("architectures") or [],
            # Transformers
            auto_model_class=transformers_info.get("auto_model"),
            processor=transformers_info.get("processor"),
            has_chat_template=bool(tokenizer_config.get("chat_template")),
            pipeline_tag=transformers_info.get("pipeline_tag") or item.get("pipeline_tag"),
            # 학습
            license=card.get("license"),
            training_datasets=datasets,
            languages=languages,
            base_model=card.get("base_model"),
            # 저장소
            model_files=model_files,
            storage_size=item.get("usedStorage"),
            spaces_count=len(spaces),
            inference_status=item.get("inference"),
            # 보안
            gated=item.get("gated"),
            security_status=sec_repo,
            security_file_details=file_security,
            # 모델 명세
            model_card=model_card,
            model_card_frontmatter=card_frontmatter,
            raw_data=item,
        )
