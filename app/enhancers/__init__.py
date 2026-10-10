from app.enhancers.base import BehaviorEnhancer, EnhanceRequest, EnhanceResult
from app.enhancers.noop import NoopBehaviorEnhancer

__all__ = [
    "BehaviorEnhancer",
    "EnhanceRequest",
    "EnhanceResult",
    "NoopBehaviorEnhancer",
    "LlmBehaviorEnhancer",
]


def __getattr__(name: str):
    # Lazy import so environments without openai/yaml can still import noop helpers.
    if name == "LlmBehaviorEnhancer":
        from app.enhancers.llm_vlm import LlmBehaviorEnhancer

        return LlmBehaviorEnhancer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
