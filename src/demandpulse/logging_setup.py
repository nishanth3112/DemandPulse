import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> None:
    # force=True so repeated calls (tests, notebooks, Streamlit reruns) don't stack handlers.
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
