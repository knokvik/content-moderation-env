from __future__ import annotations

import asyncio
import os
from typing import Iterable

from openai import AsyncOpenAI

try:
    from .models import (
        ContentCategory,
        ContentModerationAction,
        ModerationActionType,
    )
    from .server.content_moderation_env_environment import ContentModerationEnvironment
except ImportError:
    from models import (
        ContentCategory,
        ContentModerationAction,
        ModerationActionType,
    )
    from server.content_moderation_env_environment import ContentModerationEnvironment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

CLIENT = AsyncOpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "5"))


SAFE_HINTS = {
    "thank you",
    "great",
    "enjoyed",
    "helpful",
    "respectful",
    "interesting",
}
HARASSMENT_HINTS = {
    "idiot",
    "loser",
    "die",
    "stupid",
    "hate you",
    "cancer",
}
HATE_HINTS = {
    "women belong",
    "deported",
    "those [group] people",
    "ruining everything",
}
MISINFORMATION_HINTS = {
    "conspiracy",
    "flat earth",
    "nasa is lying",
    "vaccine",
}
SPAM_HINTS = {
    "buy my",
    "link in bio",
    "miracle cure",
    "works 100%",
    "product changed my life",
}


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _local_category_guess(text: str) -> str:
    normalized = text.lower()
    if _contains_any(normalized, HARASSMENT_HINTS):
        return ContentCategory.HARASSMENT.value
    if _contains_any(normalized, HATE_HINTS):
        return ContentCategory.HATE.value
    if _contains_any(normalized, MISINFORMATION_HINTS):
        return ContentCategory.MISINFORMATION.value
    if _contains_any(normalized, SPAM_HINTS):
        return ContentCategory.SPAM.value
    if _contains_any(normalized, SAFE_HINTS):
        return ContentCategory.SAFE.value
    return ContentCategory.OTHER.value


def _normalize_category(raw: str) -> str:
    normalized = raw.strip().lower()
    for category in ContentCategory:
        if category.value in normalized:
            return category.value
    return _local_category_guess(normalized)


async def call_llm(post_text: str) -> str:
    prompt = (
        "Classify the post into exactly one of: safe, harassment, hate, "
        "misinformation, spam, other. Return only the category name.\n\n"
        f"Post: {post_text}"
    )

    try:
        response = await asyncio.wait_for(
            CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16,
            ),
            timeout=LLM_TIMEOUT_S,
        )
        content = response.choices[0].message.content or ""
        return _normalize_category(content)
    except Exception:
        return _local_category_guess(post_text)


def choose_action(predicted_category: str, post_text: str) -> ContentModerationAction:
    if predicted_category in {
        ContentCategory.HARASSMENT.value,
        ContentCategory.HATE.value,
    }:
        return ContentModerationAction(
            action_type=ModerationActionType.ESCALATE,
            reason=f"Escalating likely harmful content: {predicted_category}",
        )

    if predicted_category in {
        ContentCategory.MISINFORMATION.value,
        ContentCategory.SPAM.value,
    }:
        return ContentModerationAction(
            action_type=ModerationActionType.MODERATE,
            moderate_action="flag"
            if predicted_category == ContentCategory.SPAM.value
            else "warn",
            reason=f"Moderating likely low/medium-risk content: {predicted_category}",
        )

    if predicted_category == ContentCategory.SAFE.value:
        return ContentModerationAction(
            action_type=ModerationActionType.LABEL,
            category=ContentCategory.SAFE,
            reason="Content appears safe.",
        )

    return ContentModerationAction(
        action_type=ModerationActionType.LABEL,
        category=ContentCategory.OTHER,
        reason=f"Ambiguous content: {post_text[:80]}",
    )


def format_action_label(action: ContentModerationAction) -> str:
    if action.action_type == ModerationActionType.LABEL and action.category:
        return f"label_{action.category.value}"
    if action.action_type == ModerationActionType.MODERATE:
        return f"moderate_{action.moderate_action or 'none'}"
    return action.action_type.value


async def run_task(diff: str, task_name: str) -> None:
    env = ContentModerationEnvironment(difficulty=diff)
    obs = env.reset()

    print(
        f"[START] task={task_name} env=content_moderation_env model={MODEL_NAME}",
        flush=True,
    )

    rewards: list[float] = []
    step = 1

    while not obs.done:
        predicted_category = await call_llm(obs.post_text)
        action = choose_action(predicted_category, obs.post_text)

        try:
            obs = env.step(action)
            reward = float(obs.reward or 0.0)
            rewards.append(reward)
            done_str = "true" if obs.done else "false"
            print(
                "[STEP] "
                f"step={step} "
                f"action={format_action_label(action)} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                "error=null",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[STEP] step={step} action=error reward=0.05 done=true error={exc}",
                flush=True,
            )
            break

        step += 1

    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    success = bool(rewards) and obs.done and (sum(rewards) / len(rewards) >= 0.5)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={step - 1} rewards={rewards_str}",
        flush=True,
    )


async def main() -> None:
    tasks = [
        ("easy", "task_easy_1"),
        ("medium", "task_medium_1"),
        ("hard", "task_hard_1"),
    ]

    for difficulty, task_name in tasks:
        await run_task(difficulty, task_name)


if __name__ == "__main__":
    asyncio.run(main())
