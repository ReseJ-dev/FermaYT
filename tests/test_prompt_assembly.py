"""Regression coverage for prompt transparency and desired-scene overrides."""

import asyncio
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

import app.services.visual_asset_execution as execution_module
from app.database import create_session_factory, create_sqlite_engine, init_database
from app.generators.master_scene import (
    ContinuityGenerationRequest,
    generate_continuity_image,
)
from app.models.visual_plan import VisualOperation, VisualPlan
from app.pipeline.visual_qa import VisualQAContext
from app.providers import ImageReference, ImageReferenceRole
from app.repositories import (
    create_master_scene_asset,
    get_project,
    list_beat_visual_results,
    list_master_scene_generation_attempts,
    save_project_visual_plan_record,
)
from app.services.prompt_assembly import (
    PromptOverrideState,
    PromptTargetType,
    compile_prompt_assembly,
    get_prompt_detail,
    get_prompt_sheet,
    preview_generation_request,
    semantic_requirement_for_target,
    set_prompt_override,
)
from app.services.visual_planning import hash_story_text
from tests.test_visual_asset_execution import (
    FakeImageProvider,
    FakeVisualQAClient,
    _executor,
    _qa_pass,
    _qa_regenerate,
    _setup_execution,
)


def _session(tmp_path: Path) -> tuple[Any, Any, Session]:
    engine = create_sqlite_engine(tmp_path / "prompt.db")
    init_database(engine)
    factory = create_session_factory(engine)
    return engine, factory, factory()


def test_auto_and_manual_beat_assembly_preserve_authoritative_layers(
    tmp_path: Path,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, plan, execution = _setup_execution(session)
        project = get_project(session, project_id)
        assert project is not None
        beat = plan.visual_beats[0]
        before_semantics = semantic_requirement_for_target(
            plan, PromptTargetType.BEAT, beat.id
        )
        automatic = preview_generation_request(
            session,
            project_id,
            target_type=PromptTargetType.BEAT,
            target_id=beat.id,
        )

        assert automatic.override_state == PromptOverrideState.AUTO.value
        assert automatic.manual_scene_override is None
        assert automatic.effective_scene_prompt == automatic.auto_scene_prompt
        assert automatic.final_provider_prompt
        assert "Use this permanent drawing style" in automatic.final_provider_prompt
        assert "WHAT CHANGED:" not in automatic.final_provider_prompt
        assert "VISUAL OPERATION:" not in automatic.final_provider_prompt
        assert "Target ID:" not in automatic.final_provider_prompt

        override = set_prompt_override(
            session,
            project_id,
            target_type=PromptTargetType.BEAT,
            target_id=beat.id,
            scene_prompt_override=(
                "A helmeted miner beside the collapsed underground passage"
            ),
        )
        manual = compile_prompt_assembly(
            session,
            project=project,
            plan_record=execution.visual_plan,
            plan=plan,
            target_type=PromptTargetType.BEAT,
            target_id=beat.id,
            operation=VisualOperation.REFERENCE_GENERATION,
            auto_scene_prompt=automatic.assembled_prompt_before_provider_transform,
            references=(
                ImageReference(
                    reference_id="master-1",
                    file_path="/tmp/master.png",
                    sha256="a" * 64,
                        role=ImageReferenceRole.CONTENT_CONTINUITY,
                ),
            ),
            style_id=project.style_id,
            provider="seedream",
            model="model-a",
        )

        assert manual.override_id == override.id
        assert manual.override_state == PromptOverrideState.ACTIVE.value
        assert manual.manual_scene_override == override.scene_prompt_override
        assert manual.effective_scene_prompt == override.scene_prompt_override
        assert manual.semantic_requirement == before_semantics
        qa_context = VisualQAContext.from_beat(beat)
        assert qa_context.expected_physical_state == beat.physical_state
        assert qa_context.location_id == beat.location_id
        assert manual.auto_scene_prompt != manual.effective_scene_prompt
        assert manual.operation == "REFERENCE_GENERATION"
        assert manual.operation_instructions
        assert manual.references_used[0]["reference_id"] == "master-1"
        assert "reference" in (
            manual.assembled_prompt_before_provider_transform or ""
        ).lower()
        assert "Use this permanent drawing style" in (manual.final_provider_prompt or "")
    finally:
        session.close()
        engine.dispose()


def test_manual_master_override_and_free_operations(tmp_path: Path) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, plan, _ = _setup_execution(session)
        set_prompt_override(
            session,
            project_id,
            target_type=PromptTargetType.MASTER_SCENE,
            target_id="shaft_master",
            scene_prompt_override="A clean cutaway of one stable mine shaft",
        )
        master = preview_generation_request(
            session,
            project_id,
            target_type=PromptTargetType.MASTER_SCENE,
            target_id="shaft_master",
        )
        assert master.override_state == "ACTIVE"
        assert master.effective_scene_prompt == (
            "A clean cutaway of one stable mine shaft"
        )
        assert master.semantic_requirement["environment_geometry"] == (
            plan.possible_master_scenes[0].environment_geometry
        )

        for operation, beat_id in (
            (VisualOperation.REUSE, "beat_7"),
            (VisualOperation.TRANSFORM, "beat_2"),
            (VisualOperation.OVERLAY, "beat_5"),
        ):
            set_prompt_override(
                session,
                project_id,
                target_type=PromptTargetType.BEAT,
                target_id=beat_id,
                scene_prompt_override=f"Desired future scene for {beat_id}",
            )
            assembly = preview_generation_request(
                session,
                project_id,
                target_type=PromptTargetType.BEAT,
                target_id=beat_id,
                operation=operation,
            )
            assert assembly.final_provider_prompt is None
            assert assembly.assembled_prompt_before_provider_transform is None
            assert assembly.override_state == "ACTIVE"
            assert assembly.operation_instructions
    finally:
        session.close()
        engine.dispose()


def test_provider_normalization_is_visible_and_exact(tmp_path: Path) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, _, _ = _setup_execution(session)
        set_prompt_override(
            session,
            project_id,
            target_type="MASTER_SCENE",
            target_id="shaft_master",
            scene_prompt_override="One stable cutaway shaft with fixed geometry",
        )
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override="miner " * 400,
        )
        assembly = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            provider="zimage",
            model="z-image",
        )
        assert assembly.final_provider_prompt is not None
        assert len(assembly.final_provider_prompt) <= 800
        assert any(
            item["type"] == "ZIMAGE_LIMIT_NORMALIZATION"
            for item in assembly.provider_transformations
        )
    finally:
        session.close()
        engine.dispose()


def test_real_provider_receives_snapshot_final_prompt_and_override_invalidates_cache(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine, factory, session = _session(tmp_path)
    try:
        project_id, _, execution = _setup_execution(session)
        provider = FakeImageProvider()
        monkeypatch.setattr(
            execution_module,
            "resolve_project_visual_operations",
            lambda session, project_id, provider_resolver: execution,
        )
        set_prompt_override(
            session,
            project_id,
            target_type="MASTER_SCENE",
            target_id="shaft_master",
            scene_prompt_override="One stable cutaway shaft with fixed geometry",
        )
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override="Helmeted miners at the bottom of the mine shaft",
        )
        executor = _executor(session, tmp_path, provider)
        first = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
        master_attempt = list_master_scene_generation_attempts(session, project_id)[0]
        assert master_attempt.prompt_assembly_snapshot["manual_scene_override"] == (
            "One stable cutaway shaft with fixed geometry"
        )
        assert provider.calls[0][1] == master_attempt.prompt_assembly_snapshot[
            "final_provider_prompt"
        ]
        snapshot = first.prompt_assembly_snapshot
        assert snapshot is not None
        assert provider.calls[-1][1] == snapshot["final_provider_prompt"]

        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override="Helmeted miners halfway up the same mine shaft",
        )
        second = asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
        assert second.id != first.id
        assert second.generation_revision != first.generation_revision
        assert len(provider.calls) >= 3  # one master plus two distinct beat calls
        with factory() as restarted:
            detail = get_prompt_detail(
                restarted,
                project_id,
                target_type="BEAT",
                target_id="beat_1",
            )
            assert len(detail["attempts"]) == 2
            assert detail["attempts"][0]["prompt_assembly"] == snapshot
    finally:
        session.close()
        engine.dispose()


def test_qa_retry_has_distinct_immutable_prompt_assembly(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, _, execution = _setup_execution(session)
        provider = FakeImageProvider()
        qa_client = FakeVisualQAClient(
            [
                _qa_regenerate(
                    "WRONG_PHYSICAL_STATE",
                    "The route is not blocked",
                    "Keep the composition but clearly block the passage",
                ),
                _qa_pass(),
            ]
        )
        monkeypatch.setattr(
            execution_module,
            "resolve_project_visual_operations",
            lambda session, project_id, provider_resolver: execution,
        )
        from app.pipeline.visual_qa import VisualQAService

        executor = _executor(
            session,
            tmp_path,
            provider,
            qa_service=VisualQAService(qa_client),
            max_visual_qa_attempts=2,
        )
        asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
        attempts = list_beat_visual_results(
            session,
            project_id,
            execution_plan_id=execution.id,
            beat_id="beat_1",
        )
        assert len(attempts) == 2
        first_snapshot = attempts[0].prompt_assembly_snapshot
        retry_snapshot = attempts[1].prompt_assembly_snapshot
        assert first_snapshot is not None and retry_snapshot is not None
        assert first_snapshot["qa_correction"] is None
        assert retry_snapshot["qa_correction"] == (
            "Keep the composition but clearly block the passage"
        )
        assert first_snapshot["final_provider_prompt"] != retry_snapshot[
            "final_provider_prompt"
        ]
        old_first = dict(first_snapshot)
        old_retry = dict(retry_snapshot)
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override="Helmeted miners beside the now blocked route",
        )
        asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
        after_override = list_beat_visual_results(
            session,
            project_id,
            execution_plan_id=execution.id,
            beat_id="beat_1",
        )
        assert len(after_override) == 3
        assert after_override[0].prompt_assembly_snapshot == old_first
        assert after_override[1].prompt_assembly_snapshot == old_retry
        assert after_override[0].qa_result == "REGENERATE"
        assert after_override[1].qa_result == "PASS"
        assert after_override[2].prompt_assembly_snapshot["manual_scene_override"] == (
            "Helmeted miners beside the now blocked route"
        )
        session.expire_all()
        assert attempts[0].prompt_assembly_snapshot["qa_correction"] is None
    finally:
        session.close()
        engine.dispose()


def test_revision_safety_orphaning_and_restart_persistence(tmp_path: Path) -> None:
    engine, factory, session = _session(tmp_path)
    try:
        project_id, plan, _ = _setup_execution(session)
        override = set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override="Stable override",
        )
        original_fingerprint = override.base_semantic_fingerprint

        stale_payload = plan.model_dump(mode="json")
        stale_payload["visual_strategy"] = "A revised directing strategy"
        stale_plan = VisualPlan.model_validate(stale_payload)
        project = get_project(session, project_id)
        assert project is not None
        save_project_visual_plan_record(
            session,
            project_id=project_id,
            schema_version="visual_plan_v1",
            visual_director_version="visual_director_v2",
            story_text_hash=hash_story_text(project.story_text),
            plan_json=stale_plan.model_dump(mode="json"),
        )
        stale_preview = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
        )
        assert stale_preview.override_state == "STALE"
        assert stale_preview.manual_scene_override is None

        changed_payload = stale_plan.model_dump(mode="json")
        changed_payload["visual_beats"][0]["physical_state"] = "A new flooded state"
        changed = VisualPlan.model_validate(changed_payload)
        save_project_visual_plan_record(
            session,
            project_id=project_id,
            schema_version="visual_plan_v1",
            visual_director_version="visual_director_v2",
            story_text_hash=hash_story_text(project.story_text),
            plan_json=changed.model_dump(mode="json"),
        )
        changed_preview = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
        )
        assert changed_preview.override_state == "REVIEW_REQUIRED"
        assert changed_preview.manual_scene_override is None
        assert changed_preview.semantic_fingerprint != original_fingerprint

        removed_payload = changed.model_dump(mode="json")
        replacement = removed_payload["visual_beats"][0]
        replacement["id"] = "replacement_beat"
        replacement["geography_established_by"] = None
        replacement["source_visual_id"] = None
        replacement["preferred_visual_operation"] = "NEW_IMAGE"
        removed_payload["visual_beats"] = [replacement]
        removed = VisualPlan.model_validate(removed_payload)
        save_project_visual_plan_record(
            session,
            project_id=project_id,
            schema_version="visual_plan_v1",
            visual_director_version="visual_director_v2",
            story_text_hash=hash_story_text(project.story_text),
            plan_json=removed.model_dump(mode="json"),
        )
        rows = get_prompt_sheet(session, project_id)
        orphan = next(item for item in rows if item["target_id"] == "beat_1")
        assert orphan["override_state"] == "ORPHANED"
        override_id = override.id
        session.close()

        with factory() as restarted:
            rows = get_prompt_sheet(restarted, project_id)
            persisted = next(item for item in rows if item["target_id"] == "beat_1")
            assert persisted["override_id"] == override_id
            assert persisted["override_state"] == "ORPHANED"
    finally:
        session.close()
        engine.dispose()


def test_operation_changes_keep_override_generic_and_no_provider_ops_stay_free(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, plan, execution = _setup_execution(session)
        project = get_project(session, project_id)
        assert project is not None
        desired_scene = "The miner stands beside a flooded tunnel"
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            scene_prompt_override=desired_scene,
        )
        reference = ImageReference(
            reference_id="prior-frame",
            file_path="/tmp/prior.png",
            sha256="b" * 64,
            role=ImageReferenceRole.CONTENT_CONTINUITY,
        )
        new_image = compile_prompt_assembly(
            session,
            project=project,
            plan_record=execution.visual_plan,
            plan=plan,
            target_type="BEAT",
            target_id="beat_3",
            operation="NEW_IMAGE",
            auto_scene_prompt="Automatic scene prompt",
            references=(),
            provider="seedream",
            model="model-a",
        )
        edited = compile_prompt_assembly(
            session,
            project=project,
            plan_record=execution.visual_plan,
            plan=plan,
            target_type="BEAT",
            target_id="beat_3",
            operation="EDIT_EXISTING",
            auto_scene_prompt="Automatic edit prompt",
            references=(reference,),
            provider="seedream",
            model="model-a",
        )
        assert new_image.manual_scene_override == desired_scene
        assert edited.manual_scene_override == desired_scene
        assert new_image.effective_scene_prompt == edited.effective_scene_prompt
        assert new_image.operation_instructions != edited.operation_instructions
        assert "Edit the previous image" not in desired_scene

        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_2",
            scene_prompt_override="A closer view of the same miners",
        )
        provider = FakeImageProvider()
        monkeypatch.setattr(
            execution_module,
            "resolve_project_visual_operations",
            lambda session, project_id, provider_resolver: execution,
        )
        executor = _executor(session, tmp_path, provider)
        asyncio.run(executor.execute_beat(project_id, execution.id, "beat_1"))
        paid_calls_before = len(provider.calls)
        transformed = asyncio.run(
            executor.execute_beat(project_id, execution.id, "beat_2")
        )
        assert transformed.resolved_operation == "TRANSFORM"
        assert len(provider.calls) == paid_calls_before
        persisted = get_prompt_detail(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_2",
        )
        assert persisted["stored_override"]["state"] == "ACTIVE"
    finally:
        session.close()
        engine.dispose()


def test_exact_zimage_payload_after_override_references_style_and_qa(
    tmp_path: Path,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, plan, execution = _setup_execution(session)
        project = get_project(session, project_id)
        assert project is not None
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_1",
            scene_prompt_override=(
                "Purpose: show a helmeted miner beside a broken ladder " * 30
            ),
        )
        reference = ImageReference(
            reference_id="master-reference",
            file_path="/tmp/master.png",
            sha256="c" * 64,
            role=ImageReferenceRole.CONTENT_CONTINUITY,
        )
        assembly = compile_prompt_assembly(
            session,
            project=project,
            plan_record=execution.visual_plan,
            plan=plan,
            target_type="BEAT",
            target_id="beat_1",
            operation="REFERENCE_GENERATION",
            auto_scene_prompt="Automatic visual direction",
            references=(reference,),
            provider="zimage",
            model="z-image",
            qa_correction="State: fix the missing helmet and simplify the ladder",
        )

        class Provider:
            def __init__(self) -> None:
                self.prompt: str | None = None

            async def generate_with_references(
                self,
                prompt: str,
                references: tuple[ImageReference, ...],
            ) -> str:
                assert references == (reference,)
                self.prompt = prompt
                return "fake://image"

        provider = Provider()

        async def download(url: str, output_path: str) -> str:
            assert url == "fake://image"
            return output_path

        asyncio.run(
            generate_continuity_image(
                ContinuityGenerationRequest(
                    operation=VisualOperation.REFERENCE_GENERATION,
                    prompt=assembly.final_provider_prompt or "",
                    references=(reference,),
                ),
                str(tmp_path / "candidate.png"),
                provider,  # type: ignore[arg-type]
                downloader=download,
                prompt_is_final=True,
            )
        )
        assert provider.prompt == assembly.final_provider_prompt
        assert assembly.final_provider_prompt is not None
        assert len(assembly.final_provider_prompt) <= 800
        assert assembly.qa_correction is not None
        assert assembly.manual_scene_override is not None
        assert "helmeted miner" in assembly.final_provider_prompt.lower()
        assert "fix the missing helmet" in assembly.final_provider_prompt.lower()
        forbidden_labels = (
            "Purpose:",
            "State:",
            "WHAT CHANGED:",
            "VISUAL OPERATION:",
            "Prompt Mode:",
            "Target ID:",
        )
        assert all(label not in assembly.final_provider_prompt for label in forbidden_labels)
    finally:
        session.close()
        engine.dispose()


def test_provider_resolution_and_style_change_affect_local_request_fingerprint(
    tmp_path: Path,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, _, _ = _setup_execution(session)
        desired = "The same miner stands beside knee-deep water"
        set_prompt_override(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            scene_prompt_override=desired,
        )
        seedream = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            provider="seedream",
            model="seedream-model",
        )
        zimage = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            provider="zimage",
            model="z-image",
        )
        assert seedream.operation == "EDIT_EXISTING"
        assert zimage.operation == "NEW_IMAGE"
        assert seedream.manual_scene_override == desired
        assert zimage.manual_scene_override == desired
        assert seedream.operation_instructions != zimage.operation_instructions

        project = get_project(session, project_id)
        assert project is not None
        before = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
        )
        project.global_image_style_prompt = "Use a colder muted blue palette"
        session.commit()
        after = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
        )
        assert before.semantic_fingerprint == after.semantic_fingerprint
        assert before.manual_scene_override == after.manual_scene_override == desired
        assert before.as_dict() != after.as_dict()
        assert before.final_provider_prompt != after.final_provider_prompt
    finally:
        session.close()
        engine.dispose()


def test_master_reference_change_changes_prompt_assembly_fingerprint(
    tmp_path: Path,
) -> None:
    engine, _, session = _session(tmp_path)
    try:
        project_id, _, _ = _setup_execution(session)
        project = get_project(session, project_id)
        assert project is not None
        first_path = tmp_path / "master-a.png"
        first_path.write_bytes(b"master-a")
        first_asset = create_master_scene_asset(
            session,
            project_id=project_id,
            master_scene_id="shaft_master",
            file_path=str(first_path),
            file_sha256="a" * 64,
            style_version=project.style_id,
            generation_prompt="first master",
            provider="user",
        )
        before = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            provider="seedream",
            model="seedream-model",
        )
        session.delete(first_asset)
        session.commit()
        second_path = tmp_path / "master-b.png"
        second_path.write_bytes(b"master-b")
        create_master_scene_asset(
            session,
            project_id=project_id,
            master_scene_id="shaft_master",
            file_path=str(second_path),
            file_sha256="d" * 64,
            style_version=project.style_id,
            generation_prompt="second master",
            provider="user",
        )
        after = preview_generation_request(
            session,
            project_id,
            target_type="BEAT",
            target_id="beat_3",
            provider="seedream",
            model="seedream-model",
        )
        assert before.references_used[0]["sha256"] == "a" * 64
        assert after.references_used[0]["sha256"] == "d" * 64
        assert before.as_dict() != after.as_dict()
    finally:
        session.close()
        engine.dispose()
