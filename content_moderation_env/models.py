from enum import Enum
from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ModerationActionType(str, Enum):
    LABEL = "label"
    MODERATE = "moderate"
    ESCALATE = "escalate"


class ContentCategory(str, Enum):
    SAFE = "safe"
    HARASSMENT = "harassment"
    HATE = "hate"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    OTHER = "other"


class ContentModerationAction(Action):
    action_type: ModerationActionType
    category: Optional[ContentCategory] = None
    moderate_action: Optional[Literal["remove", "warn", "flag"]] = None
    reason: Optional[str] = None


class ContentModerationObservation(Observation):
    post_text: str = ""
    thread_history: list[str] = Field(default_factory=list)
    info: dict = Field(default_factory=dict)


class ContentModerationState(State):
    current_step: int = 0
    thread_length: int = 0
    difficulty: str = "medium"
