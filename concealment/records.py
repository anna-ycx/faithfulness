"""Record what produced a set of results, saved next to them as run_info.json."""
import json
import subprocess
from pathlib import Path


def git_version() -> str:
    """Current commit hash, with '+uncommitted changes' if the working tree is dirty."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" +uncommitted changes" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_run_info(path: Path, **fields) -> None:
    info = {"code_version": git_version(), **fields}
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(f"Run info saved to {path}")
