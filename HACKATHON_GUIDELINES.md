# 🚀 Meta OpenEnv Hackathon: Strict Guidelines

> [!CAUTION]
> **CRITICAL SUBMISSION FAILURES**: Most rejections happen due to `inference.py` location, missing `HF_TOKEN`, or exceeding resource limits. Read these constraints carefully.

## 🔴 HIGH-PRIORITY CHECKLIST (MANDATORY)
- [ ] `inference.py` is in the **root directory**.
- [ ] `HF_TOKEN` is read from environment (it MUST NOT have a default value).
- [ ] `API_BASE_URL` and `MODEL_NAME` **MUST** have default values.
- [ ] Space is in **"Running"** state (not "Building").
- [ ] Container stays within **2 vCPU / 8 GB RAM**.

---

## 🏗️ 1. Runtime & Resource Constraints
Your environment will be executed in a restricted Docker container.
- **CPU**: Max **2 vCPU**
- **RAM**: Max **8 GB**
- **GPU**: No GPU guaranteed unless specified (assume CPU-only).
- **Execution**: Must build & run with `docker build` and `docker run` without custom flags.

## 🔑 2. Mandatory Environment Variables (inference.py)
The script **must** handle these exactly:

| Variable | Description | Default Requirement |
| :--- | :--- | :--- |
| `HF_TOKEN` | Hugging Face API Token | **MANDATORY (No default allowed)** |
| `API_BASE_URL` | LLM API Endpoint | **MUST have a default value** |
| `MODEL_NAME` | Model Identifier | **MUST have a default value** |

```python
# REQUIRED BOILERPLATE
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
```

## 📋 3. Functional Requirements
- **Real-World Task Simulation**: No games or toy problems. Tasks must represent real human activities (e.g., email triage, code review, data cleaning, customer support, content moderation).
- **Meaningful Reward Function**: 
    - Provide feedback throughout the trajectory (not just at the end).
    - Reward incremental progress.
    - Penalize undesirable behaviors (infinite loops, destructive actions).

## 🧰 4. OpenEnv Specification Compliance
- **Pydantic Models**: Typed `Observation`, `Action`, and `Reward` models.
- **API Interface**: `step(action)`, `reset()`, and `state()` must be fully implemented.
- **Metadata**: `openenv.yaml` file in the environment directory.
- **Validation**: Must pass `openenv validate`.

## 🎯 5. Task & Grading Requirements
- **Quantity**: Minimum of **3 tasks**.
- **Difficulty**: Easy → Medium → Hard.
- **Programmatic Graders**: 
    - Output score between `0.0` and `1.0`.
    - Deterministic, Reproducible, and Clear.

## 📤 6. Inference Output Format (STDOUT)
Exactly three line types in this order:

1. `[START] task=<task_name> env=<benchmark> model=<model_name>`
2. `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
3. `[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>`

*Rules:*
- `reward`/`rewards` → 2 decimal places.
- `done`/`success` → lowercase `true` or `false`.
- `error` → `null` if none.

## 🚀 7. Hugging Face Space Guidelines
- **Tag**: Must be tagged with `openenv`.
- **Status**: Monitor build status. Ensure it is "Running" before submitting.
- **Optimization**: Turn off unnecessary spaces to avoid resource conflicts.

---
> [!TIP]
> Use the reference projects (Calendar, Reasoning Gym, etc.) as structural guides, but focus on your unique real-world application.
