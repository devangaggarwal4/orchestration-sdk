"""hybrid-router: run LLM work locally where possible, escalate to Claude when needed."""

from hybrid_router.config import RouterConfig, load_config
from hybrid_router.decision import RouteDecision, RoutedResponse, RoutingError, Zone
from hybrid_router.router import HybridRouter
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    TextBlock,
    Tier,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "HybridRouter",
    "RouteDecision",
    "RoutedResponse",
    "RoutingError",
    "Zone",
    "CompletionRequest",
    "CompletionResponse",
    "Message",
    "RouterConfig",
    "TextBlock",
    "Tier",
    "ToolChoice",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "load_config",
]
