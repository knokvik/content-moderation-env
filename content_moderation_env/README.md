---
title: Content Moderation Environment
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - moderation
---

# Content Moderation Environment

This OpenEnv environment simulates moderation decisions over short social posts and threads.  
Each step asks the agent to label, moderate, or escalate content and returns a reward.

## Action Schema

`ContentModerationAction` fields:
- `action_type`: `label | moderate | escalate`
- `category`: `safe | harassment | hate | misinformation | spam | other` (used for `label`)
- `moderate_action`: `remove | warn | flag` (used for `moderate`)
- `reason`: optional free text

## Observation Schema

`ContentModerationObservation` fields:
- `post_text`: current content to review
- `thread_history`: previously seen posts in the same thread
- `done`: whether episode ended
- `reward`: scalar reward for latest action
- `info`: grader metadata including `correct_category` and feedback

## Reward Logic

- Rewards are normalized to the `0.0` - `1.0` range.
- Correct `label` gets full reward, while partially-correct labels get partial credit.
- `moderate` gives higher reward for better moderation choices on harmful content.
- `escalate` gets reward on medium/high severity content and zero on safe content.

## Local Run

```bash
python -m server.app --port 8000
```

## Quick Test

```bash
python -c "
from server.content_moderation_env_environment import ContentModerationEnvironment
from models import ContentModerationAction, ModerationActionType, ContentCategory
env = ContentModerationEnvironment(difficulty='hard')
obs = env.reset()
print(obs.post_text)
step = env.step(ContentModerationAction(action_type=ModerationActionType.LABEL, category=ContentCategory.HARASSMENT))
print(step.reward, step.done)
"
```

## Deploy

```bash
openenv push
```
