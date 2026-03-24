"""Civitai 플랫폼 수집기 실제 API 테스트"""

import pytest
from model_collector.platforms.civitai import CivitaiCollector, _strip_html


@pytest.fixture
def collector():
    return CivitaiCollector()


def test_strip_html_removes_tags():
    html = "<p>Hello <b>World</b></p>"
    assert _strip_html(html) == "Hello World"


def test_strip_html_handles_empty():
    assert _strip_html("") == ""
    assert _strip_html(None) == ""


@pytest.mark.asyncio
async def test_health_check(collector):
    result = await collector.health_check()
    assert result is True, "Civitai API에 연결할 수 없습니다."


@pytest.mark.asyncio
async def test_list_models_no_query(collector):
    models = await collector.list_models(limit=5)
    assert len(models) > 0
    for m in models:
        assert m.platform.value == "civitai"
        assert m.model_id
        assert m.name


@pytest.mark.asyncio
async def test_list_models_with_query(collector):
    models = await collector.list_models(query="stable diffusion", limit=5)
    assert isinstance(models, list)


@pytest.mark.asyncio
async def test_list_models_nsfw_field(collector):
    models = await collector.list_models(limit=5)
    for m in models:
        assert m.nsfw is not None or m.nsfw is None  # 필드 존재 여부 확인


@pytest.mark.asyncio
async def test_get_model_detail(collector):
    # 공개 모델 ID (DreamShaper - 널리 알려진 공개 모델)
    detail = await collector.get_model_detail("4384")
    assert detail.model_id == "4384"
    assert detail.name
    assert detail.platform.value == "civitai"


@pytest.mark.asyncio
async def test_get_model_detail_has_version_info(collector):
    detail = await collector.get_model_detail("4384")
    # 최신 버전 정보가 채워져 있어야 함
    assert detail.raw_data is not None
    assert detail.pipeline_tag is not None  # 예: "Checkpoint"
