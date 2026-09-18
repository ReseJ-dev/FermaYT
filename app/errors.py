"""Application-specific exceptions."""


class TTSGenerationError(RuntimeError):
    pass


class MediaProbeError(RuntimeError):
    """Raised when media duration cannot be determined."""


class VideoRenderError(RuntimeError):
    """Raised when FFmpeg cannot render video output."""


class ProjectTimelineRenderError(VideoRenderError):
    """Raised when a persisted ProjectTimeline cannot be rendered safely."""


class VisualDirectorError(RuntimeError):
    """Raised when a safe, validated visual plan cannot be produced."""

    def __init__(
        self,
        message: str,
        *,
        validation_category: str | None = None,
        diagnostic: dict[str, object] | None = None,
        provider_requests: int = 0,
        safe_diagnostic: object | None = None,
        user_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.validation_category = validation_category
        self.diagnostic = diagnostic
        self.provider_requests = provider_requests
        self.safe_diagnostic = safe_diagnostic
        self.user_summary = user_summary or message


class StructuredAIProviderError(RuntimeError):
    """Raised when a structured planning or vision provider call fails safely."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: object | None = None,
        user_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_diagnostic = diagnostic
        self.user_summary = user_summary or message


class PlanningBillingUncertainError(RuntimeError):
    """Pause boundary after a request may have executed remotely."""

    user_summary = (
        "Planning request timed out locally. Provider billing status is unknown. "
        "Retrying may create a second charge."
    )

    def __init__(self, attempt_id: str, outcome: str) -> None:
        super().__init__(self.user_summary)
        self.attempt_id = attempt_id
        self.outcome = outcome


class PlanningAttemptLimitError(RuntimeError):
    """Raised before dispatch when the visible paid planning budget is exhausted."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.user_summary = f"Paid planning request limit reached ({limit})"
        super().__init__(self.user_summary)


class PlanningTooLargeError(RuntimeError):
    """A planning prompt cannot fit inside its configured hard token ceilings."""

    code = "PLANNING_TOO_LARGE_FOR_SINGLE_REQUEST"

    def __init__(self, message: str) -> None:
        self.user_summary = f"{self.code}: {message}"
        super().__init__(self.user_summary)


class ProjectVisualPlanError(RuntimeError):
    """Raised when a persisted Project visual plan cannot be loaded safely."""


class StaleProjectVisualPlanError(ProjectVisualPlanError):
    """Raised when execution is attempted from an outdated semantic plan."""


class VisualOperationResolutionError(RuntimeError):
    """Raised when provider-specific visual operations cannot be resolved safely."""


class BeatVisualExecutionError(RuntimeError):
    """Raised after a beat asset execution failure has been persisted safely."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: object | None = None,
        user_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_diagnostic = diagnostic
        self.user_summary = user_summary or message


class MasterSceneError(RuntimeError):
    """Raised when an immutable master cannot be generated or verified."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: object | None = None,
        user_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_diagnostic = diagnostic
        self.user_summary = user_summary or message


class StyleContractError(ValueError):
    """Raised when an image prompt attempts to override its style contract."""


class StyleReferenceError(RuntimeError):
    """Raised when an approved style reference cannot be registered or verified."""


class VisualQAError(RuntimeError):
    """Raised when a vision model cannot return a safe structured QA decision."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: object | None = None,
        user_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_diagnostic = diagnostic
        self.user_summary = user_summary or message


class ImagePromptBuildError(ValueError):
    """Raised when semantic visual data cannot form a safe concise image prompt."""


class VideoGenerationError(RuntimeError):
    """Normalized safe error for asynchronous video generation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        provider: str | None = None,
        remote_task_id: str | None = None,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.user_summary = message
        self.provider = provider
        self.remote_task_id = remote_task_id
        self.safe_diagnostic = diagnostic or {}


VIDEO_AUTH_ERROR = "VIDEO_AUTH_ERROR"
VIDEO_BAD_REQUEST = "VIDEO_BAD_REQUEST"
VIDEO_RATE_LIMIT = "VIDEO_RATE_LIMIT"
VIDEO_SUBMISSION_TIMEOUT_UNKNOWN = "VIDEO_SUBMISSION_TIMEOUT_UNKNOWN"
VIDEO_TASK_FAILED = "VIDEO_TASK_FAILED"
VIDEO_POLL_TIMEOUT = "VIDEO_POLL_TIMEOUT"
VIDEO_DOWNLOAD_FAILED = "VIDEO_DOWNLOAD_FAILED"
VIDEO_VALIDATION_FAILED = "VIDEO_VALIDATION_FAILED"
VIDEO_BUDGET_EXCEEDED = "VIDEO_BUDGET_EXCEEDED"
VIDEO_CAPABILITY_UNSUPPORTED = "VIDEO_CAPABILITY_UNSUPPORTED"
VIDEO_PROVIDER_UNAVAILABLE = "VIDEO_PROVIDER_UNAVAILABLE"
