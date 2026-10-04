"""Settings for the local Ollama models and the workflow limits; the validator uses a second model (Week 5, Lesson 3: self-preference bias)."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "12288"))
NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "3072"))
VALIDATOR_MODEL = os.getenv("OLLAMA_VALIDATOR_MODEL", "qwen2.5:7b")


def _make_llm(temperature: float, num_predict: int, model: str = OLLAMA_MODEL) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=NUM_CTX,
        num_predict=num_predict,
    )


llm = _make_llm(0, NUM_PREDICT)
llm_retry = _make_llm(0.3, NUM_PREDICT)
llm_long = _make_llm(0, NUM_PREDICT * 2)

llm_validator = _make_llm(0, NUM_PREDICT, VALIDATOR_MODEL)
llm_validator_retry = _make_llm(0.3, NUM_PREDICT, VALIDATOR_MODEL)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RECURSION_LIMIT = 40
FIT_MAX_CONCURRENCY = int(os.getenv("FIT_MAX_CONCURRENCY", "1"))
FIT_GROUP_SIZE = int(os.getenv("FIT_GROUP_SIZE", "4"))

MAX_REQUEST_CHARS = int(os.getenv("MAX_REQUEST_CHARS", "500"))
MAX_CV_CHARS = int(os.getenv("MAX_CV_CHARS", "10000"))
MAX_JOB_CHARS = int(os.getenv("MAX_JOB_CHARS", "8000"))

MAX_JOB_RESULTS = 5
MAX_QUERY_CHARS = 100

LOG_DIR = Path(os.getenv("RUN_LOG_DIR", "logs"))
