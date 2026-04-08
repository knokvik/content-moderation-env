try:
    from .baseline_runner import main
except ImportError:
    from baseline_runner import main


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
