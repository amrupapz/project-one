"""Error types raised by the classifier."""


class ClassificationError(Exception):
    """The model call failed or its output stayed invalid after one retry."""


class MissingAPIKeyError(RuntimeError):
    """OPENROUTER_API_KEY is not set."""
