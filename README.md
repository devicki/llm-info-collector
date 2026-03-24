# Model Collector

여러 AI 모델 호스팅 플랫폼의 API를 통해 모델 메타데이터를 수집하는 Python CLI 도구입니다.

## 지원 플랫폼

| 플랫폼 | 설명 |
|--------|------|
| **Hugging Face** | 공개 모델 허브 |
| **Ollama** | 로컬 모델 서버 |
| **Civitai** | 이미지 생성 모델 커뮤니티 |

## 설치

```bash
# pyenv virtualenv 사용 시
pyenv virtualenv 3.12.3 llm-collection
pyenv local llm-collection

# 패키지 설치
pip install -e ".[dev]"
```

## 설정

```bash
cp .env.example .env
# .env 파일에 API 키 입력 (선택사항)
```

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx   # Hugging Face (없으면 anonymous rate limit)
OLLAMA_HOST=http://localhost:11434          # Ollama (기본값)
CIVITAI_API_TOKEN=your_token_here          # Civitai (없으면 공개 API만 사용)
```

## 사용법

### 플랫폼 관리

```bash
# 사용 가능한 플랫폼 목록
python -m model_collector platforms

# 전체 플랫폼 연결 상태 확인
python -m model_collector health
```

### 모델 목록 조회

```bash
python -m model_collector list --platform huggingface --query "llama" --limit 10
python -m model_collector list --platform ollama
python -m model_collector list --platform civitai --query "stable diffusion" --limit 5

# JSON 파일로 저장
python -m model_collector list --platform huggingface --query "llama" --output results.json
```

목록 테이블에 표시되는 정보: 모델 ID, 이름, 작성자, 다운로드 수, 좋아요 수, 트렌딩 점수, 태그

### 모델 상세 정보 조회

```bash
python -m model_collector detail --platform huggingface --model-id "meta-llama/Llama-3.1-8B-Instruct"
python -m model_collector detail --platform ollama --model-id "gemma3"
python -m model_collector detail --platform civitai --model-id "4384"

# JSON 파일로 저장
python -m model_collector detail --platform huggingface --model-id "meta-llama/Llama-3.1-8B-Instruct" --output detail.json

# Model Card / Modelfile 함께 출력
python -m model_collector detail --platform huggingface --model-id "meta-llama/Llama-3.1-8B-Instruct" --show-card
```

### Model Card / Modelfile 조회

모델의 명세 문서를 렌더링하여 출력합니다.

```bash
# Hugging Face — README.md(Model Card)를 Markdown으로 렌더링
python -m model_collector card --platform huggingface --model-id "google-bert/bert-base-uncased"

# Ollama — Modelfile을 구문 강조와 함께 출력
python -m model_collector card --platform ollama --model-id "gemma3"

# 원문 텍스트 파일로 저장
python -m model_collector card --platform huggingface --model-id "meta-llama/Llama-3.1-8B-Instruct" --output card.md
python -m model_collector card --platform ollama --model-id "gemma3" --output Modelfile
```

| 플랫폼 | 소스 | 렌더링 방식 |
|--------|------|------------|
| Hugging Face | `README.md` | YAML frontmatter → 테이블, 본문 → Markdown 렌더링 |
| Ollama | `modelfile` 필드 | Dockerfile 구문 강조 + 라인 번호 |

## 수집 정보 상세

### Hugging Face

| 항목 | 설명 |
|------|------|
| 아키텍처 | `model_type` + `architectures` 클래스명 (예: `LlamaForCausalLM`) |
| 파라미터 수 | safetensors 메타데이터 기반 (예: `8.0B`) |
| AutoModel 클래스 | `AutoModelForCausalLM` 등 |
| Chat Template | tokenizer에 chat template 존재 여부 |
| 저장 용량 | 전체 저장소 용량 (GB/MB) |
| 파일 목록 | 저장소 내 모든 파일 |
| 사용 Space 수 | 이 모델을 사용하는 HF Space 개수 |
| 추론 서버 상태 | `warm` / `cold` |
| 다운로드 수 | 누적 다운로드 (`downloadsAllTime`) |
| 트렌딩 점수 | 목록/상세 모두 제공 |
| 보안 스캔 | 저장소 요약 + 파일별 5개 스캐너 상세 결과 |
| Model Card | README.md 전문 (frontmatter YAML 파싱 포함) |

#### 파일별 보안 스캔 (5개 스캐너)

| 스캐너 | 제공 정보 |
|--------|---------|
| ProtectAI | 안전 여부 + 보고서 링크 |
| AV (Cisco Foundation AI) | 안티바이러스 스캔 |
| Pickle Import Scan | pickle import 목록 + 위험도 (`innocuous` / `suspicious` / `dangerous`) |
| VirusTotal | 75개 엔진 탐지 결과 |
| JFrog Research | 모델 위협 분석 |

### Ollama

| 항목 | 설명 |
|------|------|
| 파라미터 수 | `details.parameter_size` |
| 아키텍처 | `details.family` |
| 양자화 | `details.quantization_level` |
| 포맷 | 항상 `gguf` |
| 라이선스 | `license` 필드 |
| Modelfile | 모델 전체 명세 (FROM, PARAMETER, TEMPLATE 등) |

### Civitai

| 항목 | 설명 |
|------|------|
| 모델 타입 | Checkpoint, LORA, Embedding 등 |
| 베이스 모델 | SD 1.5, SDXL 1.0 등 |
| 포맷 / 양자화 | SafeTensor / fp16 등 |
| 트리거 워드 | `trainedWords` (학습 키워드) |
| NSFW 여부 | 목록/상세 모두 제공 |

## 새 플랫폼 추가

3단계만으로 새 플랫폼을 추가할 수 있습니다.

```python
# 1. platforms/new_platform.py 생성
from model_collector.core.base import BasePlatformCollector
from model_collector.core.registry import PlatformRegistry
from model_collector.core.models import ModelSummary, ModelDetail

@PlatformRegistry.register
class NewPlatformCollector(BasePlatformCollector):
    platform_name = "new_platform"
    platform_display_name = "New Platform"

    async def list_models(self, query=None, limit=20, **kwargs): ...
    async def get_model_detail(self, model_id: str): ...
    async def health_check(self): ...

# 2. platforms/__init__.py 에 import 추가
from .new_platform import NewPlatformCollector
```

## 테스트

```bash
pytest tests/ -v
```

실제 API를 호출하는 통합 테스트입니다. Ollama 테스트는 서버 미실행 시 자동으로 skip됩니다.
