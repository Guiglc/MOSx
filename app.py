import sys
import traceback
from pathlib import Path

from src.mosx_app.main import run


if __name__ == "__main__":
    try:
        run()
    except Exception:
        base_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        (base_dir / "mosx_startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
