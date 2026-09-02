"""Whole-story semantic visual planning before asset generation."""

import json
from typing import Protocol

from pydantic import ValidationError

from app.errors import VisualDirectorError
from app.models.visual_plan import VisualPlan


class VisualPlanningClient(Protocol):
    """Minimal boundary for a future structured-output LLM provider."""

    async def generate(self, prompt: str) -> str: ...


class VisualDirector:
    """Turn complete narration into a validated semantic visual plan."""

    def __init__(self, client: VisualPlanningClient) -> None:
        self._client = client

    async def create_plan(self, narration: str) -> VisualPlan:
        normalized_narration = narration.strip()
        if not normalized_narration:
            raise ValueError("Narration must not be empty")

        prompt = build_visual_director_request(normalized_narration)
        try:
            raw_plan = await self._client.generate(prompt)
        except Exception as exc:
            raise VisualDirectorError("Visual planning provider failed") from exc

        try:
            payload = json.loads(raw_plan)
            return VisualPlan.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise VisualDirectorError(
                "Visual planning provider returned an invalid structured visual plan"
            ) from exc


def build_visual_director_request(narration: str) -> str:
    """Build a provider-neutral request containing the complete narration."""
    schema = json.dumps(VisualPlan.model_json_schema(), ensure_ascii=False)
    return f"""You are the Visual Director for a coherent narrated video.

VISUAL PRODUCT PRINCIPLE. This is an automated visual storytelling system, not an
AI image generator with video export. Optimize for the experience of watching the
finished sequence. Apply this priority order to every planning trade-off:
1. storytelling clarity
2. spatial continuity
3. visual progression
4. readability
5. consistent style
6. generation efficiency
7. image beauty

A simple image that clearly advances the story is better than a beautiful image
that only decorates the narration. Do not optimize only for fewer API calls or only
for visual variety.

Read the COMPLETE narration before planning any individual beat. Narration and
visuals have different jobs: do not paraphrase narration and do not write image
generation prompts. First identify all characters, locations, important objects,
recurring environments, and possible master scenes. Then create meaningful visual
beats that make spatial layout, movement, distance, routes, obstructions, danger,
breakage, and progressive physical changes easy to understand.

MASTER SCENES. Select only a small number of important recurring environments. For
each master, define stable environment geometry, recurring object positions, overall
color palette, and basic composition. Treat masters as immutable continuity anchors,
not frames that later beats can silently replace. Every beat occurring in a recurring
environment must explicitly set master_scene_id. Descendants may change story state
such as water, damage, people, objects or lighting while preserving the recognizable
environment.

DIRECTING RULES:
1. SHOT PROGRESSION. Use WIDE, MEDIUM, CLOSE, DETAIL and CUTAWAY_DIAGRAM in a
meaningful progression. Every change of framing needs a storytelling reason; never
change camera angle merely for variety.
2. ESTABLISH GEOGRAPHY BEFORE DETAIL. Before CLOSE or DETAIL, identify an earlier
WIDE/master visual in geography_established_by that shows where the subject exists.
3. VISUAL ANTICIPATION. When useful, establish the physical setup and trajectory
before the consequence. Record what is being anticipated, without inventing events.
4. PROGRESSIVE CHANGE. Show worsening states as versions of the same environment:
ankles to knees to waist, clear to opaque, open to partially then fully blocked.
5. SCALE. Use WIDE framing when a small person versus a huge environment or machine
creates tension.
6. DISTANCE FROM SAFETY. When relevant, explicitly show current position, safe exit,
distance/scale, and the obstacle between them.
7. RESTRAIN INFOGRAPHICS. Use diagrams only when spatial understanding requires them.
Prefer the recurring illustrated environment with a simple arrow, route or highlight;
do not make the video resemble presentation slides.
8. REUSE STRONG IMAGES. Prefer REUSE, TRANSFORM, EDIT_EXISTING or OVERLAY when a
strong established frame can be zoomed, cropped, panned, edited or highlighted.
Do not request an unrelated NEW_IMAGE solely to create variety.
9. VISUAL RHYTHM. Assign duration by purpose: a brief beat may be about 2 seconds;
a strong WIDE may hold 5-7 seconds with subtle movement. Do not use one identical
duration for every beat.
10. ADD INFORMATION. Each visual must add spatial, causal, scale, route or state
information beyond the narration instead of merely illustrating its wording.

Prefer an established scene that evolves over unrelated replacement images. Target
approximately one meaningful visual change every 3-5 seconds, but do not create a new
image mechanically on that interval. Every referenced character, object, location,
and source visual must use an ID declared in this plan. A source beat must occur
earlier. Keep all descriptions semantic: this plan decides what viewers understand,
not the wording of future image prompts.

For each beat, use visual_focus for the one action/object viewers should notice first,
and must_not_show for story mistakes or distracting elements that must be excluded.

Return exactly one JSON object matching the supplied JSON Schema. Return no markdown,
code fences, commentary, or image prompts.

JSON Schema:
{schema}

COMPLETE NARRATION:
<narration>
{narration}
</narration>
"""
