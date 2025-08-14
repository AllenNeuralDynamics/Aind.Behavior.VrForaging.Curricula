import contextlib
import io
import sys
from pathlib import Path

import git

REPO_ROOT = Path(__file__).parents[1]


@contextlib.contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout


class SubmoduleManager:
    _instance = None
    initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SubmoduleManager, cls).__new__(cls)
            cls.initialized = False
        return cls._instance

    @classmethod
    def initialize_submodules(cls, force: bool = False) -> None:
        if force or not cls.initialized:
            cls._initialize_submodules()
            cls.initialized = True

    @staticmethod
    def _initialize_submodules() -> None:
        root_repo = git.Repo(REPO_ROOT)
        root_repo.git.submodule("update", "--init", "--recursive")
