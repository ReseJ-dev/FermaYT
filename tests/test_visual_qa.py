"""Tests for structured frame QA and bounded regeneration."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.errors import VisualQAError
from app.models.visual_qa import VisualQADecision, VisualQAResult
from app.pipeline.visual_qa import (
    VisualQAContext,
    VisualQAService,
    apply_visual_qa_correction,
    build_visual_qa_request,
    generate_with_visual_qa,
)


def _context(tmp_path: Path) -> VisualQAContext:
    return VisualQAContext(
        visual_purpose="Show that the mine exit is blocked",
        what_viewer_should_understand="Debris separates miners from the surface",
        required_objects=("ladder", "debris"),
        important_physical_action="The ladder breaks and debris blocks the shaft",
        location_id="main_shaft",
        expected_physical_state="Ladder missing above a debris blockage",
        style_reference_path=str(tmp_path / "STYLE_REFERENCE.png"),
        master_reference_path=str(tmp_path / "master.png"),
    )


def _pass() -> VisualQADecision:
    return VisualQADecision(
        result="PASS",
        problem_categories=[],
        reasons=[],
        correction_instruction=None,
    )


def _regenerate(
    categories: list[str],
    reason: str,
    correction: str,
) -> VisualQADecision:
    return VisualQADecision(
        result="REGENERATE",
        problem_categories=categories,
        reasons=[reason],
        correction_instruction=correction,
    )


def test_pass_cannot_hide_problems() -> None:
    with pytest.raises(ValidationError, match="PASS must not contain"):
        VisualQADecision(
            result="PASS",
            problem_categories=["COMPOSITION"],
            reasons=["Action is too small"],
            correction_instruction=None,
        )


def test_regenerate_requires_actionable_correction() -> None:
    with pytest.raises(ValidationError, match="correction_instruction"):
        VisualQADecision(
            result="REGENERATE",
            problem_categories=["STYLE_DRIFT"],
            reasons=["Walls are photorealistic"],
            correction_instruction="  ",
        )


def test_pass_with_warning_cannot_accept_a_hard_story_failure() -> None:
    with pytest.raises(ValidationError, match="hard failures"):
        VisualQADecision(
            result="PASS_WITH_WARNING",
            problem_categories=["MISSING_REQUIRED_OBJECT"],
            reasons=["The only required object is absent"],
            correction_instruction=None,
            severity="minor",
        )


def test_qa_service_sends_candidate_style_and_master_in_fixed_order(
    tmp_path: Path,
) -> None:
    class Client:
        prompt: str | None = None
        paths: tuple[str, ...] = ()

        async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
            self.prompt = prompt
            self.paths = image_paths
            return json.dumps(
                {
                    "result": "PASS",
                    "problem_categories": [],
                    "reasons": [],
                    "correction_instruction": None,
                }
            )

    client = Client()
    context = _context(tmp_path)
    decision = asyncio.run(
        VisualQAService(client).evaluate("candidate.png", context)
    )

    assert decision.result is VisualQAResult.PASS
    assert client.paths == (
        "candidate.png",
        context.style_reference_path,
        context.master_reference_path,
    )
    assert client.prompt is not None
    assert "VIDEO FRAME" in client.prompt
    assert "STORY ACCURACY" in client.prompt
    assert "CONTINUITY" in client.prompt
    assert "VIDEO READABILITY" in client.prompt
    assert "STYLE CONTRACT [rough_explainer_v1]" in client.prompt


def test_qa_compares_previous_frame_for_progression(tmp_path: Path) -> None:
    class Client:
        prompt: str | None = None
        paths: tuple[str, ...] = ()

        async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
            self.prompt = prompt
            self.paths = image_paths
            return json.dumps(
                {
                    "result": "PASS",
                    "problem_categories": [],
                    "reasons": [],
                    "correction_instruction": None,
                }
            )

    client = Client()
    context = replace(
        _context(tmp_path),
        previous_frame_path=str(tmp_path / "previous.png"),
        information_added_beyond_narration=(
            "Shows that the route now stops at the blockage"
        ),
    )

    asyncio.run(VisualQAService(client).evaluate("candidate.png", context))

    assert client.paths[-1] == context.previous_frame_path
    assert client.prompt is not None
    assert "PREVIOUS VIDEO FRAME" in client.prompt
    assert "CHECK VISUAL PROGRESSION" in client.prompt
    assert "storytelling clarity, spatial continuity" in client.prompt
    assert "Shows that the route now stops at the blockage" in client.prompt


def test_invalid_vision_json_fails_safely(tmp_path: Path) -> None:
    class Client:
        async def evaluate(self, prompt: str, image_paths: tuple[str, ...]) -> str:
            del prompt, image_paths
            return "not-json"

    with pytest.raises(VisualQAError, match="invalid structured result"):
        asyncio.run(VisualQAService(Client()).evaluate("candidate.png", _context(tmp_path)))


def test_correction_is_fed_to_regeneration_until_pass(
    tmp_path: Path,
) -> None:
    corrections: list[str | None] = []

    async def generate(correction: str | None, output_path: str) -> str:
        corrections.append(correction)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"attempt-{len(corrections)}".encode())
        return output_path

    class QA:
        calls = 0

        async def evaluate(
            self,
            image_path: str,
            context: VisualQAContext,
        ) -> VisualQADecision:
            del image_path, context
            self.calls += 1
            if self.calls == 1:
                return _regenerate(
                    ["STYLE_DRIFT", "COMPOSITION"],
                    "Rock is realistic and blockage is too small",
                    "Simplify walls and move closer to the blockage",
                )
            return _pass()

    output = tmp_path / "final.png"
    outcome = asyncio.run(
        generate_with_visual_qa(
            generate,
            str(output),
            _context(tmp_path),
            QA(),  # type: ignore[arg-type]
            max_retries=2,
        )
    )

    assert corrections == [None, "Simplify walls and move closer to the blockage"]
    assert outcome.attempts == 2
    assert outcome.decision is not None
    assert outcome.decision.result is VisualQAResult.PASS
    assert output.read_bytes() == b"attempt-2"


def test_retry_limit_keeps_best_candidate_and_records_warning(
    tmp_path: Path,
) -> None:
    decisions = [
        _regenerate(
            ["STORY_ACCURACY", "CONTINUITY"],
            "Wrong layout",
            "Restore the master layout",
        ),
        _regenerate(
            ["COMPOSITION"],
            "Object is slightly small",
            "Crop closer",
        ),
        _regenerate(
            ["STYLE_DRIFT"],
            "Too polished",
            "Use rougher lines",
        ),
    ]

    async def generate(correction: str | None, output_path: str) -> str:
        del correction
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        attempt = int(path.stem.rsplit("-", 1)[1])
        path.write_bytes(f"candidate-{attempt}".encode())
        return output_path

    class QA:
        async def evaluate(
            self,
            image_path: str,
            context: VisualQAContext,
        ) -> VisualQADecision:
            del context
            attempt = int(Path(image_path).stem.rsplit("-", 1)[1])
            return decisions[attempt - 1]

    output = tmp_path / "final.png"
    outcome = asyncio.run(
        generate_with_visual_qa(
            generate,
            str(output),
            _context(tmp_path),
            QA(),  # type: ignore[arg-type]
            max_retries=2,
        )
    )

    assert outcome.attempts == 3
    assert outcome.warning is not None
    assert "kept the best candidate" in outcome.warning
    # A slightly rougher frame is preferable to one where the story object is too
    # small to read. Clarity and readability outrank style polish.
    assert outcome.decision == decisions[2]
    assert output.read_bytes() == b"candidate-3"


def test_missing_vision_model_generates_once_and_records_warning(
    tmp_path: Path,
) -> None:
    calls = 0

    async def generate(correction: str | None, output_path: str) -> str:
        nonlocal calls
        assert correction is None
        calls += 1
        Path(output_path).write_bytes(b"candidate")
        return output_path

    output = tmp_path / "final.png"
    outcome = asyncio.run(
        generate_with_visual_qa(
            generate,
            str(output),
            _context(tmp_path),
            None,
        )
    )

    assert calls == 1
    assert outcome.decision is None
    assert outcome.warning == "Visual QA skipped: no vision-capable model configured"


def test_qa_prompt_does_not_judge_as_standalone_artwork(tmp_path: Path) -> None:
    prompt = build_visual_qa_request(_context(tmp_path))

    assert "not standalone artwork" in prompt
    assert "within a few seconds" in prompt


def test_correction_is_inserted_before_permanent_style_contract() -> None:
    from app.style_contracts import apply_image_style_contract

    original = apply_image_style_contract("A blocked mine shaft")
    corrected = apply_visual_qa_correction(
        original,
        "Remove realistic rock texture and crop closer to the blockage",
    )

    assert "VISUAL QA CORRECTION FOR REGENERATION" in corrected
    assert corrected.index("VISUAL QA CORRECTION") < corrected.index("STYLE CONTRACT [")
    assert corrected.count("STYLE CONTRACT [rough_explainer_v1]") == 1
