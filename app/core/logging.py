import logging
from contextvars import ContextVar

job_id_context: ContextVar[str] = ContextVar("job_id", default="-")


class JobContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = job_id_context.get()
        return True


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(JobContextFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s job_id=%(job_id)s %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
