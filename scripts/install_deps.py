#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def load_environment_config(config_path: Path) -> dict:
    defaults = {
        "use_uv": True,
        "requirements": "requirements.txt",
        "editable_install": True,
        "upgrade_pip": False,
    }
    if not config_path.exists():
        return defaults
    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    env = raw.get("environment", {}) or {}
    return {
        "use_uv": bool(env.get("use_uv", defaults["use_uv"])),
        "requirements": str(env.get("requirements", defaults["requirements"])).strip(),
        "editable_install": bool(env.get("editable_install", defaults["editable_install"])),
        "upgrade_pip": bool(env.get("upgrade_pip", defaults["upgrade_pip"])),
    }


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install notebook/Colab dependencies using config.toml.")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to install into")
    parser.add_argument("--force-pip", action="store_true", help="Ignore uv and use python -m pip")
    parser.add_argument("--force-uv", action="store_true", help="Use uv if available, regardless of config")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_environment_config(project_root / args.config)
    requirements = project_root / cfg["requirements"]
    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")

    if args.force_pip and args.force_uv:
        raise SystemExit("Use only one of --force-pip or --force-uv")

    use_uv = cfg["use_uv"]
    if args.force_pip:
        use_uv = False
    if args.force_uv:
        use_uv = True

    python = args.python
    uv_path = shutil.which("uv")
    can_use_uv = bool(use_uv and uv_path)

    print("Install config:")
    print(f"  python:           {python}")
    print(f"  config:           {args.config}")
    print(f"  use_uv:           {use_uv}")
    print(f"  uv found:         {uv_path or 'no'}")
    print(f"  requirements:     {requirements}")
    print(f"  editable_install: {cfg['editable_install']}")

    if cfg["upgrade_pip"] and not can_use_uv:
        run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if can_use_uv:
        cmd = ["uv", "pip", "install", "--python", python, "-r", str(requirements)]
        if cfg["editable_install"]:
            cmd += ["-e", str(project_root)]
        run(cmd)
    else:
        cmd = [python, "-m", "pip", "install", "-r", str(requirements)]
        if cfg["editable_install"]:
            cmd += ["-e", str(project_root)]
        run(cmd)


if __name__ == "__main__":
    main()
