"""Conservative capabilities exposed by a concrete image provider."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageProviderCapabilities:
    """Only operations implemented by the selected concrete client."""

    text_to_image: bool = True
    reference_generation: bool = False
    multiple_references: bool = False
    max_reference_images: int = 0
    image_editing: bool = False
    seed: bool = False
    aspect_ratio: bool = False
    supported_aspect_ratios: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        max_references = self.max_reference_images
        if self.reference_generation and max_references == 0:
            object.__setattr__(self, "max_reference_images", 1)
            max_references = 1
        if not self.reference_generation and max_references != 0:
            raise ValueError(
                "max_reference_images requires reference_generation support"
            )
        if max_references < 0:
            raise ValueError("max_reference_images must not be negative")
        if self.multiple_references and max_references < 2:
            raise ValueError(
                "multiple_references requires at least two reference images"
            )
        if self.supported_aspect_ratios and not self.aspect_ratio:
            raise ValueError(
                "supported_aspect_ratios requires aspect_ratio support"
            )

    def snapshot(self) -> dict[str, Any]:
        """Return a stable JSON-compatible capability snapshot."""
        data = asdict(self)
        data["supported_aspect_ratios"] = list(self.supported_aspect_ratios)
        return data
