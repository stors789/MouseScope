#!/usr/bin/env python3
"""Create MouseScope's optional, isolated DeepLabCut environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".venv-dlc"


def environment_python(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install DeepLabCut in a venv kept separate from MouseScope/PyInstaller."
    )
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--no-upgrade-pip", action="store_true")
    args = parser.parse_args(argv)
    env_dir = args.env.expanduser().resolve()
    print(f"Creating isolated DeepLabCut environment: {env_dir}")
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = environment_python(env_dir)
    if not args.no_upgrade_pip:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
    # DeepLabCut's official PyTorch SuperAnimal example currently recommends
    # the pre-release channel for DLC 3.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--pre", "deeplabcut"],
        check=True,
    )
    worker = ROOT / "tools" / "mousescope_dlc_worker.py"
    subprocess.run([str(python), str(worker), "status"], check=True)
    print("Ready. Restart MouseScope; .venv-dlc is discovered automatically.")
    print(
        "For another environment set MOUSESCOPE_DLC_PYTHON="
        f"{python}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
