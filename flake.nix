{
  description = "YOLOv3 chess homework: TensorFlow/Keras lesson code + Python 3.14 + uv venv + VS Codium";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python314;

        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          glib
          libGL
          libglvnd
          xorg.libX11
          xorg.libXext
          xorg.libXrender
          xorg.libxcb
          xorg.libXfixes
          xorg.libXi
          xorg.libXrandr
          xorg.libXcursor
          xorg.libXinerama
          xorg.libSM
          xorg.libICE
          freetype
          fontconfig
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            uv
            git
            wget
            curl
            unzip
            ffmpeg
            pkg-config
            vscodium
            ripgrep
            fd
          ] ++ runtimeLibs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          UV_PYTHON = "${python}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";
          UV_LINK_MODE = "copy";

          shellHook = ''
            export PROJECT_ROOT="$PWD"
            export VENV_DIR="$PROJECT_ROOT/.venv"
            export PYTHON="${python}/bin/python"
            export PYTHONNOUSERSITE=1
            export PYTHONPATH="$PROJECT_ROOT/src:''${PYTHONPATH:-}"

            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo " YOLOv3 chess homework shell"
            echo " Nix Python: $($PYTHON --version)"
            echo " uv:         $(uv --version)"
            echo " venv:       $VENV_DIR"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            if [ ! -x "$VENV_DIR/bin/python" ]; then
              echo "Creating .venv with uv + Python 3.14..."
              uv venv "$VENV_DIR" --python "$PYTHON" || echo "WARNING: could not create .venv automatically"
            fi

            export VIRTUAL_ENV="$VENV_DIR"
            export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
            export PATH="$VENV_DIR/bin:$PATH"

            echo "Active Python: $(python --version 2>/dev/null || $PYTHON --version)"
            echo "Python path:   $(which python 2>/dev/null || echo $PYTHON)"

            if [ -x "$VENV_DIR/bin/python" ] && [ -f "$PROJECT_ROOT/requirements.txt" ]; then
              REQ_HASH="$(sha256sum "$PROJECT_ROOT/requirements.txt" "$PROJECT_ROOT/pyproject.toml" 2>/dev/null | sha256sum | cut -d ' ' -f 1)"
              STAMP_FILE="$VENV_DIR/.deps.sha256"
              OLD_HASH=""
              [ -f "$STAMP_FILE" ] && OLD_HASH="$(cat "$STAMP_FILE")"

              if [ "$REQ_HASH" != "$OLD_HASH" ]; then
                echo "Installing / updating Python deps into .venv..."
                if uv pip install --python "$VENV_DIR/bin/python" -r "$PROJECT_ROOT/requirements.txt" -e "$PROJECT_ROOT"; then
                  echo "$REQ_HASH" > "$STAMP_FILE"
                  echo "Deps installed."
                else
                  echo "WARNING: dependency installation failed. You are still inside nix develop."
                  echo "Try manually: uv pip install --python .venv/bin/python -r requirements.txt -e ."
                fi
              else
                echo "Python deps are already up to date."
              fi
            fi

            echo ""
            echo "Commands:"
            echo "  python scripts/download_dataset.py"
            echo "  python -m yolo_chess.train --data chess_yolo/data.yaml --epochs 50 --batch 8 --device auto --weights auto"
            echo "  python -m yolo_chess.demo_shapes --weights runs/detect/yolov3_keras_chess/weights/best.weights.h5 --data chess_yolo/data.yaml"
            echo "  codium ."
            echo ""
          '';
        };
      }
    );
}
