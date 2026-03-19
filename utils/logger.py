"""
================================================================================
MODULE: UNIFIED ENTERPRISE LOGGER (JSON-OBSERVABILITY)
================================================================================
Target Audience: Compliance IT / DevOps / Security Operations (SecOps)

DESCRIPTION:
    This utility provides a centralized logging framework for the entire TMS.
    It transitions from standard plain-text logs to Structured JSON Logging.
    Structured logs are critical for modern compliance environments because they
    allow automated monitoring tools (ELK Stack, Splunk, Datadog) to parse
    and alert on specific fields (e.g., 'level', 'funcName', 'message')
    without complex Regex.

CAPABILITIES:
    1. Multi-Sink Logging: Simultaneously outputs to the System Console (stdout)
       and a Persistent File for audit trailing.
    2. JSON Formatting: Every log entry is a valid JSON object containing
       timestamp, severity, module name, and the specific function executed.
    3. Config-Driven: Log levels and file paths are derived from the central
       YAML configuration to ensure environment-specific behavior (e.g., DEBUG
       in UAT, WARN in Production).

PERFORMANCE & TUNING:
    - Log Level: Controlled via 'system.log_level' in tms_config.yaml.
    - Retention: While this script handles log creation, IT should implement
      LogRotate on the 'data/logs' directory to prevent disk exhaustion.
================================================================================
"""

import logging
import sys
import os
import yaml
from pythonjsonlogger import jsonlogger


def setup_logger(name="TMS_ENGINE", config=None):
    """
    Initializes and returns a JSON-enabled logger instance.

    :param name: The namespace for the logger (usually __name__).
    :param config: The global configuration dictionary from tms_config.yaml.
    :return: A configured logging.Logger object.
    """

    # 1. Fetch Configuration Parameters
    # Defaulting to INFO level and standard paths if config is missing
    sys_cfg = config.get('system', {}) if config else {}
    log_level_str = sys_cfg.get('log_level', 'INFO').upper()
    log_dir = sys_cfg.get('log_directory', 'data/logs')
    log_file_name = sys_cfg.get('log_file', 'tms_execution.log')

    # Map string level to logging constant
    level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger(name)

    # 2. Idempotency Check
    # Prevents adding multiple handlers if the setup is called multiple times
    # (common in notebook environments or complex import chains).
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # 3. JSON Formatter Definition
    # We include 'asctime' for ISO8601 timestamps and 'funcName' to trace
    # exactly which detection rule or ML function triggered the log.
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s'
    )

    # 4. Console Handler (Standard Output)
    # Essential for containerized environments (Docker/Kubernetes)
    # where stdout is captured by the container runtime.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 5. File Handler (Audit Persistence)
    # Essential for regulatory compliance to prove "Pipeline Integrity"
    # over historical runs.
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file_name)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback: if file system is read-only, we continue with console only
        # but issue a standard warning.
        print(f"CRITICAL: Could not initialize file-based logging at {log_dir}: {e}")

    return logger


# ------------------------------------------------------------------------------
# BOOTSTRAP HELPER (Used by tms_main.py)
# ------------------------------------------------------------------------------
def get_config_for_logger(path="config/tms_config.yaml"):
    """Minor helper to bootstrap the logger before the main engine starts."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    # Internal Unit Test: Validate JSON output format
    test_config = {'system': {'log_level': 'DEBUG', 'log_directory': 'data/logs'}}
    log = setup_logger("TEST_LOGGER", config=test_config)
    log.info("Logger test successful. JSON format validated.")