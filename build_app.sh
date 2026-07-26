#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

install_dependencies=true

usage() {
    cat <<'EOF'
Build MouseScope.app on macOS.

Usage:
  ./build_app.sh [--no-install]

Environment:
  MOUSESCOPE_PYTHON  Python interpreter to use. Defaults to .venv/bin/python3
                     when present, otherwise python3 from PATH.

Options:
  --no-install      Do not install/update build and runtime dependencies.
  -h, --help        Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --no-install)
            install_dependencies=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[ERROR] build_app.sh creates a macOS .app and must run on macOS." >&2
    echo "        Use build_exe.bat for the Windows build." >&2
    exit 1
fi

if [[ -n "${MOUSESCOPE_PYTHON:-}" ]]; then
    PYTHON_EXE="$MOUSESCOPE_PYTHON"
elif [[ -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
    PYTHON_EXE="$PROJECT_DIR/.venv/bin/python3"
else
    PYTHON_EXE="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_EXE:-}" || ! -x "$PYTHON_EXE" ]]; then
    echo "[ERROR] Python 3 was not found." >&2
    echo "        Set MOUSESCOPE_PYTHON to the Python interpreter to use." >&2
    exit 1
fi

echo "[INFO] Building MouseScope.app with:"
echo "       $PYTHON_EXE"

if [[ "$install_dependencies" == true ]]; then
    echo "[INFO] Installing/updating runtime and build dependencies..."
    "$PYTHON_EXE" -m pip install -r requirements.txt -r requirements-macos.txt -r requirements-build.txt
elif ! "$PYTHON_EXE" -c 'import PyInstaller' >/dev/null 2>&1; then
    echo "[ERROR] PyInstaller is not available in the selected Python environment." >&2
    echo "        Run without --no-install or install requirements-build.txt." >&2
    exit 1
fi

if ! "$PYTHON_EXE" -c 'import webview, webview.platforms.cocoa' >/dev/null 2>&1; then
    echo "[ERROR] The macOS WebView dependencies are not available." >&2
    echo "        Run without --no-install or install requirements-macos.txt." >&2
    exit 1
fi

ICON_SOURCE="$PROJECT_DIR/static/img/logo.png"
ICON_OUTPUT="$PROJECT_DIR/build/MouseScope.icns"
ICON_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mousescope-icon.XXXXXX")"
trap 'rm -rf "$ICON_WORK_DIR"' EXIT
ICONSET="$ICON_WORK_DIR/MouseScope.iconset"
mkdir -p "$ICONSET" "$PROJECT_DIR/build"

for size in 16 32 128 256 512; do
    double_size=$((size * 2))
    sips -s format png -z "$size" "$size" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -s format png -z "$double_size" "$double_size" "$ICON_SOURCE" \
        --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$ICON_OUTPUT"

"$PYTHON_EXE" -m PyInstaller --clean --noconfirm MouseScope.spec

APP_PATH="$PROJECT_DIR/dist/MouseScope.app"
APP_EXECUTABLE="$APP_PATH/Contents/MacOS/MouseScope"
if [[ ! -d "$APP_PATH" || ! -x "$APP_EXECUTABLE" ]]; then
    echo "[ERROR] Build completed without the expected app bundle: $APP_PATH" >&2
    exit 1
fi
if [[ -d "$APP_PATH/Contents/MacOS/data" ]]; then
    echo "[ERROR] Runtime data was embedded in the signed app area." >&2
    exit 1
fi
if ! /usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP_PATH/Contents/Info.plist" >/dev/null; then
    echo "[ERROR] MouseScope.app is missing a valid Info.plist." >&2
    exit 1
fi

echo
echo "[OK] Build finished:"
echo "     $APP_PATH"
echo
echo "Run with:"
echo "     open \"$APP_PATH\""
