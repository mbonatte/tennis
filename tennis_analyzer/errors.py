class AnalysisError(RuntimeError):
    """Base class for safe analysis failures."""


class InvalidVideoError(AnalysisError):
    """The uploaded file is not a supported, decodable video."""


class MissingModelError(AnalysisError):
    """A selected feature requires an unavailable model."""


class AnalysisCancelled(AnalysisError):
    """The caller requested cancellation."""


class VideoProcessingError(AnalysisError):
    """FFmpeg or OpenCV could not process the video."""
