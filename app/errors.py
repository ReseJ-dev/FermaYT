"""Application-specific exceptions."""


class TTSGenerationError(RuntimeError):
    pass


class MediaProbeError(RuntimeError):
    """Raised when media duration cannot be determined."""


class VideoRenderError(RuntimeError):
    """Raised when FFmpeg cannot render video output."""


class VisualDirectorError(RuntimeError):
    """Raised when a safe, validated visual plan cannot be produced."""


class ProjectVisualPlanError(RuntimeError):
    """Raised when a persisted Project visual plan cannot be loaded safely."""


class StaleProjectVisualPlanError(ProjectVisualPlanError):
    """Raised when execution is attempted from an outdated semantic plan."""


class VisualOperationResolutionError(RuntimeError):
    """Raised when provider-specific visual operations cannot be resolved safely."""


class BeatVisualExecutionError(RuntimeError):
    """Raised after a beat asset execution failure has been persisted safely."""


class MasterSceneError(RuntimeError):
    """Raised when an immutable master cannot be generated or verified."""


class StyleContractError(ValueError):
    """Raised when an image prompt attempts to override its style contract."""


class StyleReferenceError(RuntimeError):
    """Raised when an approved style reference cannot be registered or verified."""


class VisualQAError(RuntimeError):
    """Raised when a vision model cannot return a safe structured QA decision."""


class ImagePromptBuildError(ValueError):
    """Raised when semantic visual data cannot form a safe concise image prompt."""
