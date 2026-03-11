#!/usr/bin/env bash

BLENDER_PATH="/home/3267202/software/blender/blender"
BLENDER_PY=$($BLENDER_PATH -b --python-expr "import sys; print(sys.executable)" 2>&1 | grep -E "^/" | head -n 1)

$BLENDER_PY -m ensurepip
$BLENDER_PY -m pip install --upgrade pip
$BLENDER_PY -m pip install -r requirements.txt