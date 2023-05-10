#!/bin/sh

python -m venv --clear uetcorn_taskA_venv
# Python 3.8.5
. uetcorn_taskA_venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


deactivate
