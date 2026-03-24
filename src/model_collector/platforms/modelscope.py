import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from dotenv import load_dotenv

from ..core.base import BasePlatformCollector
from ..core.models import ModelDetail, ModelSummary, PlatformType
from ..core.registry import PlatformRegistry

load_dotenv()

_BASE_URL = "https://modelscope.cn"


@PlatformRegistry.register
class ModelScopeCollector(BasePlatformCollector):
    platform_name = "modelscope"
    platform_display_name = "ModelScope"

    def __init__(self) -> None:
        token = os.getenv("MODELSCOPE_API_TOKEN")
        headers: dict = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(base_url=_BASE_URL, headers=headers, timeout=30.0, follow_redirects=True)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        for attempt in range(3):
            resp = await self._client.request(method, path, **kwargs)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 5))
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data.get("Success", True):
                raise RuntimeError(data.get("Message", "API 오류"))
            return data
        raise RuntimeError("Rate limit 초과 - 잠시 후 다시 시도하세요.")

    async def health_check(self) -> bool:
        try:
            await self._request("PUT", "/api/v1/models", json={"Path": "", "PageNumber": 1, "PageSize": 1})
            return True
        except Exception:
            return False

    async def list_models(self, query: Optional[str] = None, limit: int = 20, **kwargs) -> list[ModelSummary]:
        # ModelScope list API: Path(소유자 필터), PageNumber, PageSize 만 지원
        body: dict = {
            "Path": query or "",
            "PageNumber": 1,
            "PageSize": limit,
        }

        try:
            data = await self._request("PUT", "/api/v1/models", json=body)
        except httpx.ConnectError:
            raise ConnectionError("ModelScope API에 연결할 수 없습니다.")

        models_list = (data.get("Data") or {}).get("Models") or []
        results: list[ModelSummary] = []
        for item in models_list:
            org = (item.get("Organization") or {}).get("Name") or item.get("Path", "")
            name = item.get("Name", "")
            model_id = f"{org}/{name}" if org else name
            tasks = item.get("Tasks") or []
            task_name = tasks[0].get("Name") if tasks else None
            tags = list(item.get("Tags") or [])
            if task_name and task_name not in tags:
                tags.insert(0, task_name)
            results.append(
                ModelSummary(
                    platform=PlatformType.MODELSCOPE,
                    model_id=model_id,
                    name=name,
                    author=org,
                    tags=tags,
                    downloads=item.get("Downloads"),
                    likes=item.get("Stars"),
                )
            )
        return results

    async def get_model_detail(self, model_id: str) -> ModelDetail:
        try:
            data = await self._request("GET", f"/api/v1/models/{model_id}")
        except httpx.ConnectError:
            raise ConnectionError("ModelScope API에 연결할 수 없습니다.")

        item = data.get("Data") or {}
        org = item.get("Organization") or {}
        tasks = item.get("Tasks") or []
        task_name = None
        for t in tasks:
            if t.get("Name"):
                task_name = t["Name"]
                break

        # 파일 정보 (safetensor)
        model_infos = item.get("ModelInfos") or {}
        safetensor = model_infos.get("safetensor") or {}
        st_files = safetensor.get("files") or []
        model_files = [f["name"] for f in st_files if "name" in f]
        storage_size = item.get("StorageSize") or (safetensor.get("model_size"))

        # 파라미터 수 추정 (model_size bytes → BF16 기준: 1 param = 2 bytes)
        parameter_count = None
        ms = safetensor.get("model_size")
        if ms:
            params = ms / 2  # BF16 assumption
            if params >= 1e9:
                parameter_count = f"{params / 1e9:.1f}B"
            elif params >= 1e6:
                parameter_count = f"{params / 1e6:.0f}M"

        # Chat template
        has_chat_template = bool(safetensor.get("chat_template"))

        # 아키텍처 하이퍼파라미터
        arch_hyperparams: dict = {}
        tensor_types = safetensor.get("tensor_type")
        if tensor_types:
            arch_hyperparams["Tensor 타입"] = ", ".join(tensor_types)
        if has_chat_template:
            arch_hyperparams["Chat Template"] = "있음"
        backend = (item.get("BackendSupport") or {}).get("backend_info") or {}
        for backend_name, label in [
            ("vllm", "vLLM"),
            ("lmdeploy_turbomind", "LMDeploy"),
            ("sglang", "SGLang"),
            ("ollama", "Ollama"),
        ]:
            info = backend.get(backend_name)
            if info and isinstance(info, dict):
                versions = ", ".join(f"{k}:{v}" for k, v in info.items() if v)
                if versions:
                    arch_hyperparams[label] = versions
        arxiv = item.get("RelatedArxivId")
        if arxiv:
            arch_hyperparams["arXiv"] = ", ".join(arxiv)
        frameworks = item.get("Frameworks")
        if frameworks:
            arch_hyperparams["프레임워크"] = ", ".join(frameworks)

        # 타임스탬프
        created = item.get("CreatedTime")
        created_at = datetime.fromtimestamp(created, tz=timezone.utc) if created else None
        updated = item.get("LastUpdatedTime")
        last_modified = datetime.fromtimestamp(updated, tz=timezone.utc) if updated else None

        # 언어
        lang = item.get("Language") or []

        # 베이스 모델
        base_models = item.get("BaseModel") or []
        base_model = ", ".join(base_models) if base_models else None

        return ModelDetail(
            platform=PlatformType.MODELSCOPE,
            model_id=model_id,
            name=item.get("Name", model_id),
            author=org.get("FullName") or org.get("Name"),
            tags=item.get("Tags") or [],
            downloads=item.get("Downloads"),
            likes=item.get("Stars"),
            last_modified=last_modified,
            created_at=created_at,
            # 아키텍처
            parameter_count=parameter_count,
            architectures=item.get("Architectures") or [],
            # Transformers
            has_chat_template=has_chat_template if safetensor else None,
            pipeline_tag=task_name,
            # 학습
            license=item.get("License"),
            languages=lang,
            base_model=base_model,
            # 저장소
            model_files=model_files,
            storage_size=storage_size,
            # 아키텍처 하이퍼파라미터
            arch_hyperparams=arch_hyperparams if arch_hyperparams else None,
            # 모델 카드 (README 이미 포함)
            model_card=item.get("ReadMeContent"),
            raw_data=item,
        )
