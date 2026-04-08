import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ContentCategory,
        ContentModerationAction,
        ContentModerationObservation,
        ContentModerationState,
        ModerationActionType,
    )
except ImportError:
    from models import (
        ContentCategory,
        ContentModerationAction,
        ContentModerationObservation,
        ContentModerationState,
        ModerationActionType,
    )


DATASET = [
    # Easy - Safe
    {"id": 1, "text": "I really enjoyed this movie, great acting!", "true_category": "safe", "severity": 0, "difficulty": "easy"},
    {"id": 2, "text": "Thank you for the helpful advice!", "true_category": "safe", "severity": 0, "difficulty": "easy"},

    # Easy - Violation
    {"id": 3, "text": "You're an idiot and should die", "true_category": "harassment", "severity": 2, "difficulty": "easy"},
    {"id": 4, "text": "Shut up you stupid loser", "true_category": "harassment", "severity": 2, "difficulty": "easy"},

    # Medium - Multi-category
    {"id": 5, "text": "All [group] people are criminals and should be deported", "true_category": "hate", "severity": 2, "difficulty": "medium"},
    {"id": 6, "text": "This vaccine is just a government conspiracy to control us", "true_category": "misinformation", "severity": 1, "difficulty": "medium"},
    {"id": 7, "text": "Buy my miracle cure for cancer, it works 100%", "true_category": "spam", "severity": 1, "difficulty": "medium"},

    # Hard - Contextual thread
    {"id": 8, "text": "This new policy is interesting", "true_category": "safe", "severity": 0, "difficulty": "hard"},
    {"id": 9, "text": "Yeah but it only helps those [group] people", "true_category": "hate", "severity": 2, "difficulty": "hard"},
    {"id": 10, "text": "Exactly, they are ruining everything", "true_category": "hate", "severity": 2, "difficulty": "hard"},

    # More realistic examples
    {"id": 11, "text": "Women belong in the kitchen", "true_category": "hate", "severity": 2, "difficulty": "medium"},
    {"id": 12, "text": "I hope you get cancer", "true_category": "harassment", "severity": 2, "difficulty": "easy"},
    {"id": 13, "text": "The earth is flat and NASA is lying", "true_category": "misinformation", "severity": 1, "difficulty": "medium"},
    {"id": 14, "text": "This product changed my life! Link in bio", "true_category": "spam", "severity": 1, "difficulty": "medium"},
    {"id": 15, "text": "Great discussion everyone, very respectful", "true_category": "safe", "severity": 0, "difficulty": "hard"},
]

class ContentModerationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    _MIN_REWARD = 0.05
    _MAX_REWARD = 0.95

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.episode_id = str(uuid.uuid4())
        self.current_step = 0
        self.ground_truth: list[dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ContentModerationObservation:
        del seed, kwargs
        self.current_step = 0
        self.episode_id = episode_id or str(uuid.uuid4())

        if self.difficulty == "easy":
            samples = [d for d in DATASET if d["difficulty"] == "easy"]
        elif self.difficulty == "hard":
            samples = [d for d in DATASET if d["difficulty"] == "hard"]
        else:
            samples = DATASET

        thread_length = min(3, len(samples)) if self.difficulty == "hard" else 1
        self.ground_truth = samples[:thread_length]

        first_post = self.ground_truth[0]["text"] if self.ground_truth else ""
        return ContentModerationObservation(
            post_text=first_post,
            thread_history=[],
            done=False,
            reward=self._MIN_REWARD,
            info={"difficulty": self.difficulty},
        )

    def step(
        self,
        action: ContentModerationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ContentModerationObservation:
        del timeout_s, kwargs
        if not self.ground_truth:
            return ContentModerationObservation(
                post_text="",
                thread_history=[],
                done=True,
                reward=self._MIN_REWARD,
                info={"error": "Environment not reset"},
            )
        if self.current_step >= len(self.ground_truth):
            return ContentModerationObservation(
                post_text="",
                thread_history=[g["text"] for g in self.ground_truth],
                done=True,
                reward=self._MIN_REWARD,
                info={"error": "Episode already finished"},
            )

        gt = self.ground_truth[self.current_step]
        raw_reward = self._score_action(action, gt)
        reward = self._normalize_reward(raw_reward)

        self.current_step += 1
        done = self.current_step >= len(self.ground_truth)
        next_text = self.ground_truth[self.current_step]["text"] if not done else ""

        return ContentModerationObservation(
            post_text=next_text,
            thread_history=[g["text"] for g in self.ground_truth[: self.current_step]],
            done=done,
            reward=reward,
            info={
                "correct_category": gt["true_category"],
                "severity": gt["severity"],
                "raw_reward": round(raw_reward, 3),
                "reward_breakdown": self._reward_breakdown(action, gt, reward),
                "grader_feedback": f"Reward: {reward:.2f}",
            },
        )

    def _score_action(self, action: ContentModerationAction, gt: dict[str, Any]) -> float:
        true_category = str(gt["true_category"])
        severity = int(gt["severity"])

        if action.action_type == ModerationActionType.LABEL:
            return self._score_label(action.category, true_category, severity)
        if action.action_type == ModerationActionType.MODERATE:
            return self._score_moderate(action.moderate_action, true_category, severity)
        if action.action_type == ModerationActionType.ESCALATE:
            return self._score_escalate(true_category, severity)
        return 0.01

    def _score_label(
        self,
        predicted: Optional[ContentCategory],
        true_category: str,
        severity: int,
    ) -> float:
        if predicted is None:
            return 0.01
        
        predicted_value = predicted.value
        if predicted_value == true_category:
            return 0.99

        true_is_safe = true_category == ContentCategory.SAFE.value
        predicted_is_safe = predicted_value == ContentCategory.SAFE.value

        # Partial credit for getting the broad safety decision right even if the
        # exact harmful category is wrong.
        if true_is_safe and not predicted_is_safe:
            return 0.25
        if not true_is_safe and not predicted_is_safe:
            return 0.6 if severity >= 1 else 0.4
        if not true_is_safe and predicted_is_safe:
            return 0.01
        return 0.2

    def _score_moderate(
        self,
        moderate_action: Optional[str],
        true_category: str,
        severity: int,
    ) -> float:
        if moderate_action is None:
            return 0.01

        if true_category == ContentCategory.SAFE.value:
            # Over-moderation is not ideal, but flagging/warning a safe post is
            # still a weakly useful intervention compared to doing nothing.
            return 0.2 if moderate_action in {"warn", "flag"} else 0.1

        if severity >= 2:
            if moderate_action == "remove":
                return 0.99
            if moderate_action == "flag":
                return 0.85
            if moderate_action == "warn":
                return 0.7
        else:
            if moderate_action == "flag":
                return 0.9
            if moderate_action == "warn":
                return 0.8
            if moderate_action == "remove":
                return 0.75

        return 0.01

    def _score_escalate(self, true_category: str, severity: int) -> float:
        if true_category == ContentCategory.SAFE.value:
            return 0.01
        if severity >= 2:
            return 0.99
        return 0.7

    def _reward_breakdown(
        self,
        action: ContentModerationAction,
        gt: dict[str, Any],
        reward: float,
    ) -> dict[str, Any]:
        return {
            "action_type": action.action_type.value,
            "true_category": gt["true_category"],
            "severity": gt["severity"],
            "reward": round(reward, 3),
        }

    @property
    def state(self) -> ContentModerationState:
        return ContentModerationState(
            episode_id=self.episode_id,
            step_count=self.current_step,
            current_step=self.current_step,
            thread_length=len(self.ground_truth),
            difficulty=self.difficulty,
        )
