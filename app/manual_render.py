"""Developer command for rendering the current persisted project timeline."""

from __future__ import annotations

import argparse

from app.database import SessionLocal, engine, init_database
from app.models.render import ProjectRenderConfig
from app.repositories import get_project, list_project_timelines
from app.services.project_render import format_render_manifest, render_project_video
from app.services.timeline import timeline_is_current


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a FermaYT ProjectTimeline")
    parser.add_argument("project_id")
    arguments = parser.parse_args()
    init_database(engine)
    with SessionLocal() as session:
        project = get_project(session, arguments.project_id)
        if project is None:
            parser.error("project was not found")
        timeline = next(
            (
                item
                for item in reversed(list_project_timelines(session, project.id))
                if timeline_is_current(session, item)
            ),
            None,
        )
        if timeline is None:
            parser.error("project has no current executable timeline")
        config = ProjectRenderConfig(
            width=project.width,
            height=project.height,
            fps=project.fps,
            image_fit_mode=project.image_fit.upper(),
        )
        result = render_project_video(session, project.id, timeline.id, config=config)
        assert result.diagnostics is not None
        print(format_render_manifest(result.diagnostics["manifest"], result.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
