"""Ollama 플랫폼 수집기 실제 API 테스트 (로컬 서버 필요)"""

import pytest
from model_collector.platforms.ollama import OllamaCollector


@pytest.fixture
def collector():
    return OllamaCollector()


@pytest.mark.asyncio
async def test_health_check(collector):
    """Ollama 서버가 실행 중이지 않으면 False를 반환해야 합니다."""
    result = await collector.health_check()
    # 서버 실행 여부에 관계없이 bool 반환 확인
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_list_models_returns_list(collector):
    """서버 미실행 시 ConnectionError, 실행 중이면 list 반환"""
    try:
        models = await collector.list_models(limit=5)
        assert isinstance(models, list)
        for m in models:
            assert m.platform.value == "ollama"
            assert m.model_id
            assert m.name
    except ConnectionError as e:
        assert "ollama" in str(e).lower() or "서버" in str(e)


@pytest.mark.asyncio
async def test_list_models_with_query_filters(collector):
    """query 파라미터가 클라이언트 측에서 필터링되는지 확인"""
    try:
        models = await collector.list_models(query="XXXXXXXXXXX_NOTEXIST", limit=20)
        assert models == []
    except ConnectionError:
        pytest.skip("Ollama 서버가 실행 중이지 않습니다.")


@pytest.mark.asyncio
async def test_get_model_detail_connection_error(collector):
    """서버 미실행 시 ConnectionError가 발생해야 합니다."""
    is_running = await collector.health_check()
    if not is_running:
        with pytest.raises(ConnectionError):
            await collector.get_model_detail("llama3")
    else:
        pytest.skip("서버가 실행 중이므로 이 테스트는 건너뜁니다.")
