"""Run the configured real one-click pipeline with stage-by-stage diagnostics."""

from __future__ import annotations

import argparse
import asyncio

from app.database import SessionLocal, engine, init_database
from app.secret_store import SecretStore
from app.services.pipeline_production import build_production_pipeline_dependencies
from app.services.project_pipeline import run_project_video_pipeline


async def _run(project_id: str) -> None:
    with SessionLocal() as session:
        dependencies = build_production_pipeline_dependencies(
            session,
            project_id,
            SecretStore(),
        )

        async def progress(
            stage: object,
            overall: int,
            stage_progress: int,
            message: str,
            current_beat: int | None,
            total_beats: int | None,
            failed_beat: str | None,
        ) -> None:
            beat = (
                f" · beat {current_beat}/{total_beats}"
                if current_beat is not None and total_beats is not None
                else ""
            )
            failure = f" · failed beat {failed_beat}" if failed_beat else ""
            print(
                f"[{overall:3d}%] {getattr(stage, 'value', stage)} "
                f"({stage_progress:3d}%): {message}{beat}{failure}",
                flush=True,
            )

        report = await run_project_video_pipeline(
            session,
            project_id,
            dependencies,
            progress=progress,
        )
        print("\nPIPELINE REPORT")
        for key, value in report.as_dict().items():
            print(f"{key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FermaYT automated pipeline with configured real providers"
    )
    parser.add_argument("project_id")
    arguments = parser.parse_args()
    init_database(engine)
    asyncio.run(_run(arguments.project_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
