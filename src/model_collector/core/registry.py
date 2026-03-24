from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BasePlatformCollector


class PlatformRegistry:
    """
    플랫폼 수집기를 자동 등록/관리합니다.
    새 플랫폼 추가 시 @PlatformRegistry.register 데코레이터만 붙이면 됩니다.
    """

    _collectors: dict[str, type[BasePlatformCollector]] = {}

    @classmethod
    def register(cls, collector_class: type[BasePlatformCollector]) -> type[BasePlatformCollector]:
        cls._collectors[collector_class.platform_name] = collector_class
        return collector_class

    @classmethod
    def get_collector(cls, platform_name: str) -> BasePlatformCollector:
        if platform_name not in cls._collectors:
            available = ", ".join(cls._collectors.keys())
            raise ValueError(
                f"알 수 없는 플랫폼: '{platform_name}'. 사용 가능한 플랫폼: {available}"
            )
        return cls._collectors[platform_name]()

    @classmethod
    def list_platforms(cls) -> list[str]:
        return list(cls._collectors.keys())

    @classmethod
    def list_all(cls) -> list[type[BasePlatformCollector]]:
        return list(cls._collectors.values())
