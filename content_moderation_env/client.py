from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import (
    ContentModerationAction,
    ContentModerationObservation,
    ContentModerationState,
)


class ContentModerationEnv(
    EnvClient[
        ContentModerationAction,
        ContentModerationObservation,
        ContentModerationState,
    ]
):
    def _step_payload(self, action: ContentModerationAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[ContentModerationObservation]:
        obs = ContentModerationObservation(**payload.get("observation", {}))
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> ContentModerationState:
        return ContentModerationState(**payload)
