"""Dataset -> ``Sample(prompt, expected)`` converters (see base.py for the framework and output layout,
sources.py / templates.py for the bases, mix.py for frozen eval mixes)."""

from .base import CONVERTERS, BaseConverter, get_converter, register
from .sources import (
    HFDatasetSource,
    HFFilesSource,
    LocalFileSource,
    MultiConfigHFSource,
)
from .templates import (
    PromptLabelConverter,
    SafeOnlyConverter,
    TextLabelConverter,
    UnsafeOnlyConverter,
)

__all__ = [
    "CONVERTERS",
    "BaseConverter",
    "HFDatasetSource",
    "HFFilesSource",
    "LocalFileSource",
    "MultiConfigHFSource",
    "PromptLabelConverter",
    "SafeOnlyConverter",
    "TextLabelConverter",
    "UnsafeOnlyConverter",
    "get_converter",
    "register",
]
