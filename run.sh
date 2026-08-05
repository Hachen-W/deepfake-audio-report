#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec env -u LD_LIBRARY_PATH -u GTK_PATH -u QT_PLUGIN_PATH python app.py
