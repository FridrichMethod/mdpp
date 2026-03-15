#!/bin/bash

set -euo pipefail

# conda create -n mdpp python=3.12 -y
# conda activate mdpp

conda install -c conda-forge pdbfixer -y

pip install uv
uv pip install -e ".[viz,dev,mypy,docs]"
