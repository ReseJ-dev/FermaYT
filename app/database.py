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
    _migrate_project_visual_plans_to_scoped_storage(engine)
    _recover_interrupted_planning_attempts(engine)
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
            "planning_budget_amount": "NUMERIC(18, 8)",
            "planning_max_paid_requests": "INTEGER NOT NULL DEFAULT 2",
            "planning_max_input_tokens": "INTEGER NOT NULL DEFAULT 20000",
            "planning_max_output_tokens": "INTEGER NOT NULL DEFAULT 32768",
            "planning_max_total_estimated_tokens": "INTEGER NOT NULL DEFAULT 100000",
            "visual_qa_enabled": "BOOLEAN NOT NULL DEFAULT 1",
            "visual_qa_provider": "VARCHAR(50) NOT NULL DEFAULT 'dashscope'",
            "visual_qa_model": "VARCHAR(255) NOT NULL DEFAULT 'qwen-vl-max'",
            "style_id": "VARCHAR(64) NOT NULL DEFAULT 'rough_explainer_v1'",
            "generation_budget_enabled": "BOOLEAN NOT NULL DEFAULT 0",
            "generation_budget_amount": "NUMERIC(18, 8)",
            "generation_budget_currency": "VARCHAR(3) NOT NULL DEFAULT 'EUR'",
            "generation_budget_warning_threshold": "FLOAT NOT NULL DEFAULT 0.8",
            "draft_video_path": "TEXT",
            "draft_rendered_at": "DATETIME",
            "draft_paid_visual_ratio": "FLOAT NOT NULL DEFAULT 0.3",
            "draft_width": "INTEGER NOT NULL DEFAULT 1280",
            "draft_height": "INTEGER NOT NULL DEFAULT 720",
            "pilot_video_path": "TEXT",
            "pilot_rendered_at": "DATETIME",
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
    if "project_visual_execution_plans" in table_names:
        execution_columns = {
            column["name"]
            for column in inspector.get_columns("project_visual_execution_plans")
        }
        execution_additions = {
            "production_profile": "VARCHAR(16) NOT NULL DEFAULT 'FINAL'",
            "production_profile_version": "VARCHAR(64) NOT NULL DEFAULT 'final_v1'",
        }
        with engine.begin() as connection:
            for name, sql_type in execution_additions.items():
                if name not in execution_columns:
                    connection.execute(text(
                        f"ALTER TABLE project_visual_execution_plans "
                        f"ADD COLUMN {name} {sql_type}"
                    ))
    if "beat_visual_results" in table_names:
        beat_profile_columns = {
            column["name"]
            for column in inspector.get_columns("beat_visual_results")
        }
        beat_additions = {
            "production_profile": "VARCHAR(16) NOT NULL DEFAULT 'FINAL'",
            "asset_role": "VARCHAR(32) NOT NULL DEFAULT 'RENDERABLE_BEAT'",
            "prompt_assembly_snapshot": "JSON",
        }
        if any(name not in beat_profile_columns for name in beat_additions):
            with engine.begin() as connection:
                for name, sql_type in beat_additions.items():
                    if name not in beat_profile_columns:
                        connection.execute(text(
                            "ALTER TABLE beat_visual_results ADD COLUMN "
                            f"{name} {sql_type}"
                        ))
    if "project_video_renders" in table_names:
        render_columns = {
            column["name"]
            for column in inspector.get_columns("project_video_renders")
        }
        render_additions = {
            "production_profile": "VARCHAR(16) NOT NULL DEFAULT 'FINAL'",
            "generation_scope_type": "VARCHAR(32) NOT NULL DEFAULT 'FULL'",
            "generation_scope_value": "FLOAT",
        }
        with engine.begin() as connection:
            for name, sql_type in render_additions.items():
                if name in render_columns:
                    continue
                connection.execute(text(
                    f"ALTER TABLE project_video_renders ADD COLUMN {name} {sql_type}"
                ))
    if "project_timelines" in table_names:
        timeline_columns = {
            column["name"] for column in inspector.get_columns("project_timelines")
        }
        timeline_additions = {
            "generation_scope_type": "VARCHAR(32) NOT NULL DEFAULT 'FULL'",
            "generation_scope_value": "FLOAT",
            "generation_scope_version": (
                "VARCHAR(64) NOT NULL DEFAULT 'generation_scope_v1'"
            ),
        }
        with engine.begin() as connection:
            for name, sql_type in timeline_additions.items():
                if name not in timeline_columns:
                    connection.execute(text(
                        f"ALTER TABLE project_timelines ADD COLUMN {name} {sql_type}"
                    ))
    if "project_visual_plans" in table_names:
        columns = {
            column["name"]
            for column in inspector.get_columns("project_visual_plans")
        }
        plan_additions = {
            "story_text_hash": "VARCHAR(64)",
            "scope_key": "VARCHAR(64) NOT NULL DEFAULT 'FULL'",
            "scope_type": "VARCHAR(32) NOT NULL DEFAULT 'FULL'",
            "requested_seconds": "FLOAT",
            "requested_beats": "INTEGER",
            "source_start_char": "INTEGER NOT NULL DEFAULT 0",
            "source_end_char": "INTEGER NOT NULL DEFAULT 0",
            "source_hash": "VARCHAR(64) NOT NULL DEFAULT ''",
            "is_partial": "BOOLEAN NOT NULL DEFAULT 0",
            "continuation_boundary_char": "INTEGER",
            "expected_beat_count": "INTEGER",
            "planning_max_output_tokens": "INTEGER",
        }
        with engine.begin() as connection:
            for name, sql_type in plan_additions.items():
                if name in columns:
                    continue
                connection.execute(
                    text(f"ALTER TABLE project_visual_plans ADD COLUMN {name} {sql_type}")
                )
    if "planning_provider_attempts" in table_names:
        attempt_columns = {
            column["name"]
            for column in inspector.get_columns("planning_provider_attempts")
        }
        attempt_additions = {
            "expected_beat_count": "INTEGER",
            "cost_certainty": "VARCHAR(16) NOT NULL DEFAULT 'UNKNOWN'",
            "progress_state": "VARCHAR(40) NOT NULL DEFAULT 'WAITING_FOR_PROVIDER'",
            "validation_category": "VARCHAR(100)",
            "dispatched_at": "DATETIME",
            "response_received_at": "DATETIME",
            "validation_started_at": "DATETIME",
        }
        with engine.begin() as connection:
            for name, sql_type in attempt_additions.items():
                if name in attempt_columns:
                    continue
                connection.execute(text(
                    "ALTER TABLE planning_provider_attempts "
                    f"ADD COLUMN {name} {sql_type}"
                ))
    if "master_scene_assets" in table_names:
        master_columns = {
            column["name"] for column in inspector.get_columns("master_scene_assets")
        }
        if "prompt_assembly_snapshot" not in master_columns:
            with engine.begin() as connection:
                connection.execute(text(
                    "ALTER TABLE master_scene_assets "
                    "ADD COLUMN prompt_assembly_snapshot JSON"
                ))

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


def _migrate_project_visual_plans_to_scoped_storage(engine: Engine) -> None:
    """Replace the legacy one-plan-per-project constraint without losing plan IDs."""
    inspector = inspect(engine)
    if "project_visual_plans" not in inspector.get_table_names():
        return
    legacy_unique = any(
        constraint.get("column_names") == ["project_id"]
        for constraint in inspector.get_unique_constraints("project_visual_plans")
    )
    if not legacy_unique:
        return
    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.executescript(
            """
            CREATE TABLE project_visual_plans_scoped (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                project_id VARCHAR(36) NOT NULL,
                schema_version VARCHAR(32) NOT NULL,
                visual_director_version VARCHAR(32) NOT NULL,
                story_text_hash VARCHAR(64) NOT NULL,
                scope_key VARCHAR(64) NOT NULL DEFAULT 'FULL',
                scope_type VARCHAR(32) NOT NULL DEFAULT 'FULL',
                requested_seconds FLOAT,
                requested_beats INTEGER,
                source_start_char INTEGER NOT NULL DEFAULT 0,
                source_end_char INTEGER NOT NULL DEFAULT 0,
                source_hash VARCHAR(64) NOT NULL DEFAULT '',
                is_partial BOOLEAN NOT NULL DEFAULT 0,
                continuation_boundary_char INTEGER,
                expected_beat_count INTEGER,
                planning_max_output_tokens INTEGER,
                plan_json JSON NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                CONSTRAINT uq_project_visual_plans_project_scope
                    UNIQUE (project_id, scope_key),
                FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE
            );
            INSERT INTO project_visual_plans_scoped (
                id, project_id, schema_version, visual_director_version,
                story_text_hash, scope_key, scope_type, requested_seconds,
                requested_beats, source_start_char, source_end_char, source_hash,
                is_partial, continuation_boundary_char, expected_beat_count,
                planning_max_output_tokens, plan_json, created_at, updated_at
            ) SELECT
                id, project_id, schema_version, visual_director_version,
                COALESCE(story_text_hash, ''), 'FULL', 'FULL', NULL, NULL, 0,
                source_end_char, source_hash, 0, NULL, expected_beat_count,
                planning_max_output_tokens, plan_json, created_at, updated_at
            FROM project_visual_plans;
            DROP TABLE project_visual_plans;
            ALTER TABLE project_visual_plans_scoped RENAME TO project_visual_plans;
            CREATE INDEX ix_project_visual_plans_project_id
                ON project_visual_plans (project_id);
            """
        )
        connection.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        connection.close()


def _recover_interrupted_planning_attempts(engine: Engine) -> None:
    """Never interpret an interrupted paid POST as a free failed request."""
    if "planning_provider_attempts" not in inspect(engine).get_table_names():
        return
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE planning_provider_attempts "
                "SET status = 'INTERRUPTED_BILLING_UNKNOWN', "
                "billing_status = 'UNKNOWN', "
                "progress_state = 'PAUSED_AFTER_TIMEOUT', "
                "remote_execution_status = 'REMOTE_EXECUTION_STATUS_UNKNOWN', "
                "safe_error = 'Application restarted while provider attempt was pending', "
                "finished_at = CURRENT_TIMESTAMP "
                "WHERE status = 'PENDING'"
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
