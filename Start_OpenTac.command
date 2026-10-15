#!/bin/bash
# OpenTac CAD on this computer (macOS / Linux).
#   macOS: double-click this file.   Linux: bash Start_OpenTac.command
# The first run downloads what the app needs (a few minutes, once). Later runs start in seconds.
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "  OpenTac needs uv (a small Python installer, https://docs.astral.sh/uv/)."
    read -r -p "  Install it now? [y/N] " answer
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        echo "  Install uv, then run this file again."
        read -r -p "  Press Enter to close. " _
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

exec uv run opentac cad
