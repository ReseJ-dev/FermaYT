"""SQLAlchemy configuration for local SQLite persistence."""

import sqlite3
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_PATH = Path("data/app.db")


class Base(DeclarativeBase):
    pass


def create_sqlite_engine(database_path: str | Path) -> Engine:
    """Create a SQLite engine with foreign-key enforcement enabled."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(
        dbapi_connection: sqlite3.Connection,
        connection_record: object,
    ) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to an engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_database(engine: Engine) -> None:
    """Create all known tables when they do not exist."""
    from app import persistence  # noqa: F401

    Base.metadata.create_all(engine)
    _apply_additive_schema_updates(engine)
    _recover_interrupted_beat_visual_results(engine)
    _recover_interrupted_video_renders(engine)


def _apply_additive_schema_updates(engine: Engine) -> None:
    """Keep the pre-release local SQLite schema compatible without Alembic."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "projects" in table_names:
        project_columns = {
            column["name"] for column in inspector.get_columns("projects")
        }
        project_additions = {
            "planning_provider": "VARCHAR(50) NOT NULL DEFAULT 'dashscope'",
            "planning_model": "VARCHAR(255) NOT NULL DEFAULT 'qwen-plus'",
            "visual_qa_enabled": "BOOLEAN NOT NULL DEFAULT 1",
            "visual_qa_provider": "VARCHAR(50) NOT NULL DEFAULT 'dashscope'",
            "visual_qa_model": "VARCHAR(255) NOT NULL DEFAULT 'qwen-vl-max'",
            "style_id": "VARCHAR(64) NOT NULL DEFAULT 'rough_explainer_v1'",
        }
        missing_project_columns = {
            name: sql_type
            for name, sql_type in project_additions.items()
            if name not in project_columns
        }
        if missing_project_columns:
            with engine.begin() as connection:
                for name, sql_type in missing_project_columns.items():
                    connection.execute(
                        text(f"ALTER TABLE projects ADD COLUMN {name} {sql_type}")
                    )
    if "project_visual_plans" in table_names:
        columns = {
            column["name"]
            for column in inspector.get_columns("project_visual_plans")
        }
        if "story_text_hash" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE project_visual_plans "
                        "ADD COLUMN story_text_hash VARCHAR(64)"
                    )
                )

    if "beat_visual_results" not in table_names:
        return
    beat_columns = {
        column["name"] for column in inspector.get_columns("beat_visual_results")
    }
    qa_columns = {
        "qa_result": "VARCHAR(32)",
        "qa_scores": "JSON",
        "qa_problem_categories": "JSON",
        "qa_reasons": "JSON",
        "qa_correction_instruction": "TEXT",
        "qa_provider": "VARCHAR(100)",
        "qa_model": "VARCHAR(255)",
        "qa_attempt": "INTEGER",
        "qa_revision": "VARCHAR(64)",
        "qa_prompt_version": "VARCHAR(64)",
        "qa_warning": "TEXT",
        "accepted_at": "DATETIME",
        "manual_qa_override": "VARCHAR(32)",
        "manual_qa_reason": "TEXT",
        "manual_qa_at": "DATETIME",
    }
    missing = [
        (name, sql_type)
        for name, sql_type in qa_columns.items()
        if name not in beat_columns
    ]
    if missing:
        with engine.begin() as connection:
            for name, sql_type in missing:
                connection.execute(
                    text(
                        f"ALTER TABLE beat_visual_results "
                        f"ADD COLUMN {name} {sql_type}"
                    )
                )


def _recover_interrupted_beat_visual_results(engine: Engine) -> None:
    """Make pre-restart PENDING beat attempts explicitly retryable."""
    if "beat_visual_results" not in inspect(engine).get_table_names():
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE beat_visual_results "
                "SET generation_status = 'FAILED', "
                "error = 'Beat execution interrupted by application restart', "
                "is_accepted = 0, updated_at = CURRENT_TIMESTAMP "
                "WHERE generation_status = 'PENDING'"
            )
        )


def _recover_interrupted_video_renders(engine: Engine) -> None:
    """Make interrupted local FFmpeg attempts explicit and retryable."""
    if "project_video_renders" not in inspect(engine).get_table_names():
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE project_video_renders "
                "SET status = 'FAILED', "
                "safe_error = 'Render interrupted by application restart', "
                "failed_stage = 'INTERRUPTED', "
                "completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE status = 'PENDING'"
            )
        )


engine = create_sqlite_engine(DATABASE_PATH)
SessionLocal = create_session_factory(engine)
