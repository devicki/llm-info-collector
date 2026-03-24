from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class PlatformType(str, Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CIVITAI = "civitai"


class ModelSummary(BaseModel):
    """모델 목록 조회 시 반환되는 요약 정보"""

    platform: PlatformType
    model_id: str
    name: str
    author: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    downloads: Optional[int] = None
    likes: Optional[int] = None
    trending_score: Optional[float] = None      # 목록 API에서도 제공
    last_modified: Optional[datetime] = None
    created_at: Optional[datetime] = None
    nsfw: Optional[bool] = None


class ModelDetail(ModelSummary):
    """특정 모델의 상세 정보"""

    # 모델 아키텍처 정보
    parameter_count: Optional[str] = None
    architecture: Optional[str] = None          # model_type (예: "llama", "bert")
    architectures: list[str] = Field(default_factory=list)  # 구체적 클래스 (예: ["LlamaForCausalLM"])
    quantization: Optional[str] = None
    format: Optional[str] = None
    arch_hyperparams: Optional[dict] = None     # config.json 기반 하이퍼파라미터

    # Transformers 정보
    auto_model_class: Optional[str] = None      # 예: "AutoModelForCausalLM"
    processor: Optional[str] = None             # 예: "AutoTokenizer"
    has_chat_template: Optional[bool] = None    # chat_template 존재 여부
    pipeline_tag: Optional[str] = None

    # 학습 정보
    license: Optional[str] = None
    training_datasets: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    base_model: Optional[str] = None

    # 저장소 정보
    model_files: list[str] = Field(default_factory=list)    # 저장소 파일 목록
    storage_size: Optional[int] = None          # 바이트 단위 저장 용량
    spaces_count: Optional[int] = None          # 이 모델을 사용하는 Space 수
    inference_status: Optional[str] = None      # "warm" / "cold"

    # 안전성 / 보안
    security_status: Optional[dict] = None          # 저장소 수준 요약 (scansDone, filesWithIssues)
    security_file_details: list[dict] = Field(default_factory=list)  # 파일별 상세 스캔 결과
    gated: Optional[bool] = None

    # 모델 명세 문서
    model_card: Optional[str] = None           # HF: README.md 전문 / Ollama: Modelfile 전문
    model_card_frontmatter: Optional[dict] = None  # HF YAML frontmatter 파싱 결과

    # 플랫폼 원본 데이터
    raw_data: Optional[dict] = None
