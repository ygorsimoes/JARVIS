from __future__ import annotations

import logging
import os
import sys

import structlog
from structlog.types import Processor


def _shared_processors(*, include_formatted_exceptions: bool) -> list[Processor]:
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if include_formatted_exceptions:
        processors.insert(-1, structlog.processors.format_exc_info)
    return processors


def configure_logging(level: str = "INFO", log_format: str = "console") -> None:
    renderer: Processor
    use_json: bool = bool(
        log_format == "json" or (log_format == "auto" and os.getenv("CI"))
    )
    processors = _shared_processors(include_formatted_exceptions=use_json)
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    formatter_processors: list[Processor] = [
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        renderer,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=processors,
        processors=formatter_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level.upper())

    for logger_name in (
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    configured_processors: list[Processor] = [
        *processors,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]
    structlog.configure(
        processors=configured_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    return structlog.stdlib.get_logger(name)


def bind_context(**context: str) -> None:
    structlog.contextvars.bind_contextvars(**context)
