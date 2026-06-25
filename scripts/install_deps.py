#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Literal


TF_SENSITIVE_PREFIXES = (
    "tensorflow",
    "tf-nightly",
    "keras",
    "keras-nightly",
    "tensorboard",
    "tb-nightly",
)


def detect_colab() -> bool:
    """Return True only for a real Google Colab runtime."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    if "google.colab" in sys.modules:
        return True
    if importlib.util.find_spec("google.colab") is not None and Path("/content").exists():
        return True
    return False


def running_from_nix_venv() -> bool:
    exe = Path(sys.executable).as_posix()
    return "/nix/store/" in exe or "/.venv/bin/python" in exe or os.environ.get("IN_NIX_SHELL") is not None


def current_python_has_pip() -> bool:
    return importlib.util.find_spec("pip") is not None


def load_environment_config(config_path: Path) -> dict:
    defaults = {
        # "auto" -> real Colab uses pip if available; local/Nix does not try to install.
        # "pip"  -> use pip internal API in the current Python only.
        # "none" -> do not install anything.
        # "uv"   -> not supported in no-subprocess mode; use external uv command manually.
        "install_backend": "auto",
        "use_uv": None,
        "requirements": "requirements-colab.txt",
        "editable_install": True,
        "upgrade_pip": False,
        "skip_tensorflow_on_colab": True,
        "editable_no_deps_on_colab": True,
    }
    if not config_path.exists():
        return defaults

    with config_path.open("rb") as f:
        raw = tomllib.load(f)
    env = raw.get("environment", {}) or {}

    if "install_backend" in env:
        backend = str(env.get("install_backend", defaults["install_backend"])).strip().lower()
    elif "use_uv" in env:
        # Backward compatibility only. In no-subprocess mode uv cannot be called from this script.
        backend = "uv" if bool(env.get("use_uv")) else "pip"
    else:
        backend = defaults["install_backend"]

    if backend not in {"auto", "pip", "uv", "none"}:
        raise ValueError("[environment].install_backend must be one of: auto, pip, uv, none")

    return {
        "install_backend": backend,
        "use_uv": env.get("use_uv", defaults["use_uv"]),
        "requirements": str(env.get("requirements", defaults["requirements"])).strip(),
        "editable_install": bool(env.get("editable_install", defaults["editable_install"])),
        "upgrade_pip": bool(env.get("upgrade_pip", defaults["upgrade_pip"])),
        "skip_tensorflow_on_colab": bool(env.get("skip_tensorflow_on_colab", defaults["skip_tensorflow_on_colab"])),
        "editable_no_deps_on_colab": bool(env.get("editable_no_deps_on_colab", defaults["editable_no_deps_on_colab"])),
    }


def filtered_requirements_for_colab(requirements: Path) -> Path:
    safe_lines: list[str] = []
    skipped: list[str] = []

    for line in requirements.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped or stripped.startswith("#"):
            safe_lines.append(line)
            continue
        if lowered.startswith(TF_SENSITIVE_PREFIXES):
            skipped.append(line)
            continue
        safe_lines.append(line)

    if skipped:
        print("Colab detected: skipping TensorFlow/Keras-sensitive packages:")
        for line in skipped:
            print(f"  - {line}")

    tmp = tempfile.NamedTemporaryFile("w", suffix="-requirements-colab-safe.txt", delete=False, encoding="utf-8")
    with tmp:
        tmp.write("\n".join(safe_lines).rstrip() + "\n")
    return Path(tmp.name)


def choose_backend(config_backend: str, *, is_colab: bool, pip_ok: bool, force_pip: bool, force_none: bool) -> Literal["pip", "none"]:
    if force_pip and force_none:
        raise SystemExit("Use only one of --force-pip or --force-none")
    if force_none:
        return "none"
    if force_pip:
        return "pip"

    if config_backend == "none":
        return "none"
    if config_backend == "uv":
        print("WARNING: install_backend='uv' is not supported by scripts/install_deps.py anymore because this script does not use subprocess.")
        print("         Use an external command instead: uv pip install --python .venv/bin/python -r requirements.txt -e .")
        return "none"
    if config_backend == "auto":
        if is_colab and pip_ok:
            return "pip"
        # Local Nix/Jupyter environments should already be prepared by flake.nix.
        return "none"
    if config_backend == "pip":
        return "pip" if pip_ok else "none"
    raise AssertionError(config_backend)


def run_pip(args: list[str]) -> None:
    """Run pip without subprocess, using pip's internal CLI in the current interpreter."""
    try:
        from pip._internal.cli.main import main as pip_main
    except Exception as exc:
        raise RuntimeError("Current Python does not have importable pip") from exc

    print("+ python -m pip", " ".join(args), flush=True)
    code = pip_main(args)
    if code:
        raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install deps without subprocess; Colab-safe and Nix-safe.")
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--force-pip", action="store_true", help="Use pip internal API in current Python")
    parser.add_argument("--force-none", action="store_true", help="Do not install anything")
    parser.add_argument("--print-env", action="store_true", help="Print detected environment and exit")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_environment_config(project_root / args.config)
    requirements = project_root / cfg["requirements"]
    is_colab = detect_colab()
    is_nix = running_from_nix_venv()
    pip_ok = current_python_has_pip()
    backend = choose_backend(
        cfg["install_backend"],
        is_colab=is_colab,
        pip_ok=pip_ok,
        force_pip=args.force_pip,
        force_none=args.force_none,
    )

    print("Install config:")
    print(f"  python:                 {sys.executable}")
    print(f"  config:                 {args.config}")
    print(f"  detected_colab:         {is_colab}")
    print(f"  detected_nix_or_venv:   {is_nix}")
    print(f"  install_backend config: {cfg['install_backend']}")
    print(f"  chosen_backend:         {backend}")
    print(f"  python_has_pip:         {pip_ok}")
    print(f"  requirements:           {requirements}")
    print(f"  editable_install:       {cfg['editable_install']}")

    if args.print_env or backend == "none":
        if backend == "none":
            print("No dependency installation performed.")
            if is_nix:
                print("Local/Nix note: use nix develop / flake shell, or run uv manually outside this Python process.")
        return

    if not requirements.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements}")

    install_req = requirements
    if is_colab and cfg["skip_tensorflow_on_colab"]:
        install_req = filtered_requirements_for_colab(requirements)

    if not pip_ok:
        print("Current Python has no pip; skipping install instead of crashing.")
        return

    if cfg["upgrade_pip"]:
        run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])

    run_pip(["install", "-r", str(install_req)])

    if cfg["editable_install"]:
        editable_args = ["install", "-e", str(project_root)]
        if is_colab and cfg["editable_no_deps_on_colab"]:
            editable_args.append("--no-deps")
        run_pip(editable_args)


if __name__ == "__main__":
    main()
