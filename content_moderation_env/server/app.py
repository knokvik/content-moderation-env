try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ContentModerationAction, ContentModerationObservation
    from .content_moderation_env_environment import ContentModerationEnvironment
except (ImportError, ModuleNotFoundError):
    from models import ContentModerationAction, ContentModerationObservation
    from server.content_moderation_env_environment import ContentModerationEnvironment


app = create_app(
    ContentModerationEnvironment,
    ContentModerationAction,
    ContentModerationObservation,
    env_name="content_moderation_env",
    max_concurrent_envs=1,
)


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    host = args.host
    port = args.port
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
