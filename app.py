"""FastAPI application root re-export."""

from realtime_inference.app import app, create_app, main

__all__ = ["app", "create_app", "main"]

if __name__ == "__main__":
    main()
