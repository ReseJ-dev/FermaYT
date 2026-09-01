"""Safe local paths for project media assets."""

import re
from pathlib import Path
from uuid import uuid4

_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]+$")


class ProjectMediaPaths:
    """Build and create paths inside one project's media directory."""

    def __init__(
        self,
        project_id: str,
        projects_root: str | Path = "data/projects",
    ) -> None:
        self.project_id = _safe_component(project_id, "project_id")
        self.projects_root = Path(projects_root).resolve()
        self._project_dir = self.projects_root / self.project_id
        for directory in (
            self.images_dir,
            self.master_scenes_dir,
            self.audio_dir,
            self.scenes_dir,
            self.uploads_dir,
            self.output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    @property
    def images_dir(self) -> Path:
        return self.project_dir / "images"

    @property
    def master_scenes_dir(self) -> Path:
        return self.project_dir / "master_scenes"

    @property
    def audio_dir(self) -> Path:
        return self.project_dir / "audio"

    @property
    def scenes_dir(self) -> Path:
        return self.project_dir / "scenes"

    @property
    def uploads_dir(self) -> Path:
        return self.project_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.project_dir / "output"

    def image_path(self, scene_id: str) -> Path:
        scene_name = _safe_component(scene_id, "scene_id")
        return self.images_dir / f"{scene_name}.png"

    def master_scene_path(self, master_scene_id: str, style_version: str) -> Path:
        master_name = _safe_component(master_scene_id, "master_scene_id")
        version = _safe_component(style_version, "style_version")
        return self.master_scenes_dir / f"{master_name}-{version}.png"

    def audio_path(self, scene_id: str, extension: str = ".wav") -> Path:
        scene_name = _safe_component(scene_id, "scene_id")
        if not _SAFE_EXTENSION.fullmatch(extension):
            raise ValueError("invalid audio file extension")
        return self.audio_dir / f"{scene_name}{extension.lower()}"

    def scene_video_path(self, scene_id: str) -> Path:
        scene_name = _safe_component(scene_id, "scene_id")
        return self.scenes_dir / f"{scene_name}.mp4"

    def uploaded_image_path(self, extension: str = ".png") -> Path:
        if not _SAFE_EXTENSION.fullmatch(extension):
            raise ValueError("invalid upload file extension")
        return self.uploads_dir / f"{uuid4()}{extension.lower()}"

    @property
    def final_video_path(self) -> Path:
        return self.output_dir / "final.mp4"


def _safe_component(value: str, field_name: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f"invalid {field_name}")
    return value
