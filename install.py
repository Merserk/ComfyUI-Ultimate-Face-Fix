from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.model_setup.manager import run_manager_hook


if __name__ == "__main__":
    raise SystemExit(run_manager_hook())
