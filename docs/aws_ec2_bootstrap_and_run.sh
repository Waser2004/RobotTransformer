#!/usr/bin/env bash
set -euo pipefail

# Run this from a fresh Ubuntu EC2 instance after the repository has already
# been cloned or copied onto the machine.
#
# Optional overrides:
#   REPO_DIR=/home/ubuntu/RobotTransformer
#   INSTALL_BLENDER=0
#   BLENDER_BIN=/custom/path/to/blender

REPO_DIR="${REPO_DIR:-$HOME/RobotTransformer}"
INSTALL_BLENDER="${INSTALL_BLENDER:-1}"

# Install system packages required for Python, headless X11, video encoding,
# and Blender's common Linux runtime dependencies.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3 \
  python3-venv \
  python3-pip \
  git \
  xvfb \
  xauth \
  ffmpeg \
  libgl1 \
  libxi6 \
  libxrender1 \
  libxkbcommon-x11-0 \
  libsm6 \
  libxext6 \
  libxxf86vm1

# Install Blender from Ubuntu packages unless the instance already has a
# working Blender install and you plan to set BLENDER_BIN manually.
if [[ "${INSTALL_BLENDER}" == "1" ]]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y blender
fi

cd "${REPO_DIR}"

# Create and activate the project virtual environment.
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies used by expert data generation.
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Prefer an explicit BLENDER_BIN override, otherwise use Blender from PATH.
if [[ -z "${BLENDER_BIN:-}" ]]; then
  BLENDER_BIN="$(command -v blender || true)"
  export BLENDER_BIN
fi

if [[ -z "${BLENDER_BIN:-}" ]]; then
  echo "Blender was not found. Install Blender and export BLENDER_BIN before running again."
  exit 1
fi

mkdir -p logs

echo "Using Blender at: ${BLENDER_BIN}"
"${BLENDER_BIN}" --version | head -n 1 || true

# Start the Blender-backed generator in a virtual display so it can run on a
# headless EC2 machine.
python3 src/expert_data_generation/generate_with_blender.py \
  --use-xvfb always \
  --blender-log-file logs/blender.log

# Useful follow-up commands in a second shell:
#   tail -f logs/blender.log
#   ss -ltn | grep 5055
