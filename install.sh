#!/usr/bin/env bash
# macOS / Linux installer. Requires Python 3.10+, FFmpeg and Deno (or Node.js) to be installed.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
command -v ffmpeg >/dev/null || echo "Note: install FFmpeg (brew install ffmpeg / sudo apt install ffmpeg)."
command -v deno >/dev/null || command -v node >/dev/null || echo "Note: install Deno (https://deno.com) for YouTube downloads."
echo "Done. Run ./start.sh"
