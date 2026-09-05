"""Application-level exceptions mapped to HTTP responses in main.py."""
from __future__ import annotations


class ValidationFailedError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class RateLimitExceededError(Exception):
    pass
