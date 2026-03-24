from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax

from ..core.models import ModelSummary, ModelDetail

console = Console()


def _fmt_num(n: int | None) -> str:
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def display_platforms(platforms: list[tuple[str, str]]) -> None:
    table = Table(title="사용 가능한 플랫폼", box=box.ROUNDED)
    table.add_column("ID", style="cyan bold")
    table.add_column("이름", style="white")
    for pid, pname in platforms:
        table.add_row(pid, pname)
    console.print(table)


def display_health(results: list[tuple[str, str, bool]]) -> None:
    table = Table(title="플랫폼 헬스체크", box=box.ROUNDED)
    table.add_column("플랫폼", style="cyan bold")
    table.add_column("표시 이름", style="white")
    table.add_column("상태", justify="center")
    for pid, pname, ok in results:
        status = Text("✓ 정상", style="green bold") if ok else Text("✗ 연결 실패", style="red bold")
        table.add_row(pid, pname, status)
    console.print(table)


def display_models_table(models: list[ModelSummary], platform: str) -> None:
    if not models:
        console.print(f"[yellow]'{platform}' 플랫폼에서 모델을 찾을 수 없습니다.[/yellow]")
        return

    table = Table(
        title=f"{platform} 모델 목록 ({len(models)}개)",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("모델 ID", style="cyan", max_width=45, overflow="fold")
    table.add_column("이름", style="white", max_width=35, overflow="fold")
    table.add_column("작성자", style="green", max_width=20)
    table.add_column("다운로드", justify="right", style="yellow")
    table.add_column("좋아요", justify="right", style="magenta")
    table.add_column("트렌딩", justify="right", style="red")
    table.add_column("태그", max_width=30, overflow="fold")

    for m in models:
        tags_str = ", ".join(m.tags[:5]) if m.tags else "-"
        trending = f"{m.trending_score:.0f}" if m.trending_score is not None else "-"
        table.add_row(
            m.model_id,
            m.name,
            m.author or "-",
            _fmt_num(m.downloads),
            _fmt_num(m.likes),
            trending,
            tags_str,
        )
    console.print(table)


_STATUS_STYLE = {
    "safe": "green",
    "unscanned": "yellow",
    "dangerous": "red bold",
}


def _status_text(status: str | None) -> Text:
    s = (status or "unknown").lower()
    style = _STATUS_STYLE.get(s, "white")
    return Text(s, style=style)


def _display_file_security(details: list[dict]) -> None:
    table = Table(
        title="파일별 보안 스캔 상세",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("파일", style="cyan", max_width=40, overflow="fold")
    table.add_column("종합", justify="center")
    table.add_column("ProtectAI", justify="center")
    table.add_column("AV (Cisco)", justify="center")
    table.add_column("Pickle", justify="center")
    table.add_column("VirusTotal", justify="center")
    table.add_column("JFrog", justify="center")
    table.add_column("Pickle 임포트", max_width=35, overflow="fold")

    def _scanner_cell(entry: dict, key: str) -> Text | str:
        """스캐너 키가 없으면 '-', 있으면 status 텍스트 반환."""
        if key not in entry:
            return "-"
        return _status_text((entry[key] or {}).get("status"))

    for f in details:
        pkl = f.get("pickle_scan") or {}

        # pickle imports 요약: safety != innocuous 인 것만 강조
        imports = pkl.get("pickleImports") or []
        suspicious = [
            f"{i['module']}.{i['name']}({i['safety']})"
            for i in imports
            if i.get("safety") != "innocuous"
        ]
        if suspicious:
            pkl_imports_str = ", ".join(suspicious)
        elif imports:
            pkl_imports_str = f"{len(imports)}개 (모두 안전)"
        elif pkl.get("message"):
            pkl_imports_str = pkl["message"]
        else:
            pkl_imports_str = "-"

        table.add_row(
            f.get("path", ""),
            _status_text(f.get("overall_status")),
            _scanner_cell(f, "protect_ai"),
            _scanner_cell(f, "av_scan"),
            _scanner_cell(f, "pickle_scan"),
            _scanner_cell(f, "virustotal"),
            _scanner_cell(f, "jfrog"),
            pkl_imports_str,
        )
    console.print(table)


def display_model_detail(model: ModelDetail) -> None:
    lines: list[str] = []

    def _add(label: str, value) -> None:
        if value is not None and value != [] and value != {}:
            lines.append(f"[bold cyan]{label}:[/bold cyan] {value}")

    def _fmt_storage(n: int | None) -> str | None:
        if n is None:
            return None
        if n >= 1 << 30:
            return f"{n / (1 << 30):.1f} GB"
        if n >= 1 << 20:
            return f"{n / (1 << 20):.1f} MB"
        return f"{n / (1 << 10):.1f} KB"

    _add("플랫폼", model.platform.value)
    _add("모델 ID", model.model_id)
    _add("이름", model.name)
    _add("작성자", model.author)
    _add("설명", (model.description or "")[:200] + ("..." if model.description and len(model.description) > 200 else "") or None)
    # 아키텍처
    _add("아키텍처", model.architecture)
    _add("세부 아키텍처", ", ".join(model.architectures) if model.architectures else None)
    _add("파라미터", model.parameter_count)
    _add("양자화", model.quantization)
    _add("포맷", model.format)
    # Transformers
    _add("AutoModel 클래스", model.auto_model_class)
    _add("프로세서", model.processor)
    _add("Chat Template", "있음" if model.has_chat_template else ("없음" if model.has_chat_template is not None else None))
    _add("파이프라인", model.pipeline_tag)
    # 학습
    _add("라이선스", model.license)
    _add("베이스 모델", model.base_model)
    _add("언어", ", ".join(model.languages) if model.languages else None)
    _add("학습 데이터셋", ", ".join(model.training_datasets[:10]) if model.training_datasets else None)
    _add("태그", ", ".join(model.tags[:10]) if model.tags else None)
    # 저장소
    _add("저장 용량", _fmt_storage(model.storage_size))
    _add("파일 수", str(len(model.model_files)) if model.model_files else None)
    _add("사용 Space 수", str(model.spaces_count) if model.spaces_count else None)
    _add("추론 서버", model.inference_status)
    # 통계
    _add("다운로드", _fmt_num(model.downloads))
    _add("좋아요", _fmt_num(model.likes))
    _add("트렌딩 점수", str(model.trending_score) if model.trending_score is not None else None)
    # 보안
    _add("접근 제한", str(model.gated) if model.gated is not None else None)
    _add("NSFW", str(model.nsfw) if model.nsfw is not None else None)
    if model.security_status:
        scanned = model.security_status.get("scansDone")
        issues = model.security_status.get("filesWithIssues") or []
        _add("보안 스캔", f"완료={scanned}, 이슈 파일={len(issues)}개")


    content = "\n".join(lines)
    console.print(Panel(content, title=f"[bold white]{model.name}[/bold white]", border_style="blue"))

    if model.arch_hyperparams:
        table = Table(title="아키텍처 하이퍼파라미터 (config.json)", box=box.ROUNDED, show_lines=True)
        table.add_column("항목", style="cyan bold", min_width=20)
        table.add_column("값", style="white")
        for k, v in model.arch_hyperparams.items():
            table.add_row(k, str(v))
        console.print(table)

    if model.model_versions:
        vtable = Table(title="버전 목록", box=box.ROUNDED, show_lines=True)
        vtable.add_column("버전", style="cyan bold", max_width=25, overflow="fold")
        vtable.add_column("베이스 모델", style="white", max_width=20)
        vtable.add_column("다운로드", justify="right", style="yellow")
        vtable.add_column("파일 수", justify="right", style="green")
        vtable.add_column("출시일", style="magenta", max_width=22)
        for v in model.model_versions:
            pub = (v.get("published_at") or "")[:10]  # YYYY-MM-DD
            vtable.add_row(
                v.get("name", "-"),
                v.get("base_model", "-"),
                _fmt_num(v.get("downloads")),
                str(v.get("file_count", 0)),
                pub or "-",
            )
        console.print(vtable)

    if model.security_file_details:
        _display_file_security(model.security_file_details)


def display_model_card(model: ModelDetail) -> None:
    """Model Card(HuggingFace README.md) 또는 Modelfile(Ollama)을 렌더링합니다."""
    if not model.model_card:
        console.print("[yellow]모델 명세 문서가 없습니다.[/yellow]")
        return

    platform = model.platform.value

    if platform == "huggingface":
        # YAML frontmatter 테이블 출력
        fm = model.model_card_frontmatter
        if fm and not fm.get("_raw"):
            table = Table(title="Model Card Frontmatter", box=box.ROUNDED, show_lines=True)
            table.add_column("키", style="cyan bold", max_width=25)
            table.add_column("값", style="white", max_width=70, overflow="fold")
            for k, v in fm.items():
                val = ", ".join(v) if isinstance(v, list) else str(v)
                table.add_row(k, val)
            console.print(table)

        # frontmatter 제거 후 Markdown 본문 렌더링
        import re
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", model.model_card, flags=re.DOTALL)
        console.print(
            Panel(
                Markdown(body),
                title=f"[bold white]Model Card — {model.name}[/bold white]",
                border_style="green",
            )
        )

    elif platform == "civitai":
        console.print(
            Panel(
                model.model_card,
                title=f"[bold white]모델 설명 — {model.name}[/bold white]",
                border_style="green",
            )
        )

    elif platform == "ollama":
        # Modelfile은 Docker-like DSL → bash 구문강조로 가독성 확보
        console.print(
            Panel(
                Syntax(model.model_card, "dockerfile", theme="monokai", line_numbers=True),
                title=f"[bold white]Modelfile — {model.name}[/bold white]",
                border_style="green",
            )
        )

    elif platform == "modelscope":
        console.print(
            Panel(
                Markdown(model.model_card),
                title=f"[bold white]Model Card — {model.name}[/bold white]",
                border_style="green",
            )
        )
