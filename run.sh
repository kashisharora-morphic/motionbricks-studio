#!/usr/bin/env bash
cd "$(dirname "$0")"
exec .venv/bin/python server/server.py --port "${PORT:-8765}" "$@"
