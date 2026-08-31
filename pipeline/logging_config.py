import logging
import os

# On a machine behind a corporate TLS-inspecting proxy (confirmed here:
# Bekaert's network runs Zscaler, whose root CA is trusted by Windows but
# NOT by Python's bundled certifi list -- every plain urllib/requests HTTPS
# call fails CERTIFICATE_VERIFY_FAILED as a result, found via a real
# features/news.py failure; yfinance itself was unaffected only because its
# curl_cffi dependency already uses the OS-native trust store). Patches
# Python's ssl module to verify against the OS certificate store instead of
# certifi -- the standard fix for this exact scenario, not a verification
# bypass. Imported here (this module is pulled in by virtually every other
# module in the project via get_logger) so it's active before any network
# call happens, regardless of entry point. No-op/harmless on a machine
# without this problem.
import truststore
truststore.inject_into_ssl()

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "logs")


def get_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
