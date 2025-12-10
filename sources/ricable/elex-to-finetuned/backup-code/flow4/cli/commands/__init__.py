"""CLI command implementations."""

from .pipeline import PipelineCommand
from .convert import ConvertCommand
from .chunk import ChunkCommand
from .chat_command import ChatCommand
from .optimize_command import OptimizeCommand

# Optional commands
try:
    from .finetune import FinetuneCommand
    _FINETUNE_AVAILABLE = True
except ImportError:
    _FINETUNE_AVAILABLE = False
    FinetuneCommand = None

try:
    from .generate import GenerateCommand
    _GENERATE_AVAILABLE = True
except ImportError:
    _GENERATE_AVAILABLE = False
    GenerateCommand = None

try:
    from .deduplicate import DeduplicateCommand  
    _DEDUPLICATE_AVAILABLE = True
except ImportError:
    _DEDUPLICATE_AVAILABLE = False
    DeduplicateCommand = None

__all__ = [
    "PipelineCommand",
    "ConvertCommand", 
    "ChunkCommand",
    "ChatCommand",
    "OptimizeCommand",
]

if _FINETUNE_AVAILABLE:
    __all__.append("FinetuneCommand")

if _GENERATE_AVAILABLE:
    __all__.append("GenerateCommand")

if _DEDUPLICATE_AVAILABLE:
    __all__.append("DeduplicateCommand")