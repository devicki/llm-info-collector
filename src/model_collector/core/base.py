from abc import ABC, abstractmethod
from typing import Optional
from .models import ModelSummary, ModelDetail


class BasePlatformCollector(ABC):
    """
    모든 플랫폼 수집기가 구현해야 하는 인터페이스.
    새 플랫폼 추가 시 이 클래스를 상속하고 아래 메서드를 구현하면 됩니다.
    """

    platform_name: str
    platform_display_name: str

    @abstractmethod
    async def list_models(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        **kwargs,
    ) -> list[ModelSummary]:
        """모델 목록을 검색/조회합니다."""
        ...

    @abstractmethod
    async def get_model_detail(self, model_id: str) -> ModelDetail:
        """특정 모델의 상세 정보를 가져옵니다."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """플랫폼 API 연결 상태를 확인합니다."""
        ...
