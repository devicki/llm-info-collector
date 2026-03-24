"""Hugging Face 플랫폼 수집기 실제 API 테스트"""

import pytest
from model_collector.platforms.huggingface import HuggingFaceCollector


@pytest.fixture
def collector():
    return HuggingFaceCollector()


@pytest.mark.asyncio
async def test_health_check(collector):
    result = await collector.health_check()
    assert result is True, "Hugging Face API에 연결할 수 없습니다."


@pytest.mark.asyncio
async def test_list_models_no_query(collector):
    models = await collector.list_models(limit=5)
    assert len(models) > 0
    for m in models:
        assert m.platform.value == "huggingface"
        assert m.model_id
        assert m.name


@pytest.mark.asyncio
async def test_list_models_with_query(collector):
    models = await collector.list_models(query="llama", limit=5)
    assert len(models) > 0
    # 검색 결과에 'llama'가 포함되어야 함
    names_and_ids = " ".join(m.name.lower() + " " + m.model_id.lower() for m in models)
    assert "llama" in names_and_ids


@pytest.mark.asyncio
async def test_list_models_returns_model_summary_fields(collector):
    models = await collector.list_models(limit=3)
    m = models[0]
    assert m.downloads is not None or m.downloads is None  # 필드 존재 여부만 확인
    assert isinstance(m.tags, list)


@pytest.mark.asyncio
async def test_get_model_detail(collector):
    detail = await collector.get_model_detail("google-bert/bert-base-uncased")
    assert detail.model_id
    assert detail.name
    assert detail.platform.value == "huggingface"


@pytest.mark.asyncio
async def test_get_model_detail_has_extra_fields(collector):
    detail = await collector.get_model_detail("google-bert/bert-base-uncased")
    # raw_data가 채워져 있어야 함
    assert detail.raw_data is not None
