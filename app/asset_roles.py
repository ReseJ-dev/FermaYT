"""Explicit roles controlling whether a visual asset may enter a final timeline."""

from enum import Enum


class VisualAssetRole(str, Enum):
    RENDERABLE_BEAT = "RENDERABLE_BEAT"
    STYLE_REFERENCE = "STYLE_REFERENCE"
    MASTER_ONLY = "MASTER_ONLY"
    DEBUG = "DEBUG"
    OPERATION_REFERENCE = "OPERATION_REFERENCE"


RENDERABLE_VISUAL_ASSET_ROLES = frozenset({VisualAssetRole.RENDERABLE_BEAT.value})
