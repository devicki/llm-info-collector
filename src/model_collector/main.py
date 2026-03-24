import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv
from rich.console import Console

from . import platforms as _platforms_module  # noqa: F401 — trigger registration
from .core.registry import PlatformRegistry
from .utils.display import (
    display_health,
    display_model_card,
    display_model_detail,
    display_models_table,
    display_platforms,
)

load_dotenv()

console = Console()


# ---------------------------------------------------------------------------
# async command handlers
# ---------------------------------------------------------------------------


async def cmd_platforms() -> None:
    collectors = PlatformRegistry.list_all()
    items = [(c.platform_name, c.platform_display_name) for c in collectors]
    display_platforms(items)


async def cmd_health() -> None:
    collectors = PlatformRegistry.list_all()
    results = []
    for cls in collectors:
        collector = cls()
        ok = await collector.health_check()
        results.append((cls.platform_name, cls.platform_display_name, ok))
    display_health(results)


async def cmd_list(platform: str, query: str | None, limit: int, output: str | None) -> None:
    collector = PlatformRegistry.get_collector(platform)
    try:
        models = await collector.list_models(query=query, limit=limit)
    except ConnectionError as e:
        console.print(f"[red]연결 오류:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]오류 발생:[/red] {e}")
        sys.exit(1)

    display_models_table(models, platform)

    if output:
        data = [m.model_dump(mode="json") for m in models]
        with open(output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"[green]결과가 '{output}' 파일에 저장되었습니다.[/green]")


async def cmd_detail(platform: str, model_id: str, output: str | None, show_card: bool = False) -> None:
    collector = PlatformRegistry.get_collector(platform)
    try:
        detail = await collector.get_model_detail(model_id)
    except ConnectionError as e:
        console.print(f"[red]연결 오류:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]오류 발생:[/red] {e}")
        sys.exit(1)

    display_model_detail(detail)

    if show_card:
        display_model_card(detail)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(detail.model_dump(mode="json"), f, ensure_ascii=False, indent=2, default=str)
        console.print(f"[green]결과가 '{output}' 파일에 저장되었습니다.[/green]")


async def cmd_card(platform: str, model_id: str, output: str | None) -> None:
    collector = PlatformRegistry.get_collector(platform)
    try:
        detail = await collector.get_model_detail(model_id)
    except ConnectionError as e:
        console.print(f"[red]연결 오류:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]오류 발생:[/red] {e}")
        sys.exit(1)

    display_model_card(detail)

    if output and detail.model_card:
        with open(output, "w", encoding="utf-8") as f:
            f.write(detail.model_card)
        console.print(f"[green]모델 명세가 '{output}' 파일에 저장되었습니다.[/green]")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-collector",
        description="LLM 모델 메타데이터 수집 CLI 도구",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # platforms
    sub.add_parser("platforms", help="사용 가능한 플랫폼 목록 출력")

    # health
    sub.add_parser("health", help="전체 플랫폼 헬스체크")

    # list
    p_list = sub.add_parser("list", help="모델 목록 조회")
    p_list.add_argument("--platform", "-p", required=True, help="플랫폼 ID (huggingface/ollama/civitai/modelscope)")
    p_list.add_argument("--query", "-q", default=None, help="검색어")
    p_list.add_argument("--limit", "-l", type=int, default=20, help="최대 결과 수 (기본: 20)")
    p_list.add_argument("--output", "-o", default=None, help="JSON 파일로 저장 경로")

    # detail
    p_detail = sub.add_parser("detail", help="모델 상세 정보 조회")
    p_detail.add_argument("--platform", "-p", required=True, help="플랫폼 ID")
    p_detail.add_argument("--model-id", "-m", required=True, help="모델 ID")
    p_detail.add_argument("--output", "-o", default=None, help="JSON 파일로 저장 경로")
    p_detail.add_argument("--show-card", action="store_true", help="Model Card / Modelfile 함께 출력")

    # card
    p_card = sub.add_parser("card", help="Model Card(HF) 또는 Modelfile(Ollama) 출력")
    p_card.add_argument("--platform", "-p", required=True, help="플랫폼 ID")
    p_card.add_argument("--model-id", "-m", required=True, help="모델 ID")
    p_card.add_argument("--output", "-o", default=None, help="원문 텍스트 파일로 저장 경로")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "platforms":
        asyncio.run(cmd_platforms())
    elif args.command == "health":
        asyncio.run(cmd_health())
    elif args.command == "list":
        asyncio.run(cmd_list(args.platform, args.query, args.limit, args.output))
    elif args.command == "detail":
        asyncio.run(cmd_detail(args.platform, args.model_id, args.output, args.show_card))
    elif args.command == "card":
        asyncio.run(cmd_card(args.platform, args.model_id, args.output))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
