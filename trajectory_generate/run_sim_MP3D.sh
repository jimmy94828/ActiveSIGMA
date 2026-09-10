#!/usr/bin/env bash
set -euo pipefail

# Habitat-Sim rendering follows the same OpenGL/X setup used by the ActiveSGM
# main scripts. Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export GL_BACKEND="${GL_BACKEND:-x11}"

GLDISPATCH=/lib/x86_64-linux-gnu/libGLdispatch.so.0
if [[ -f "${GLDISPATCH}" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${GLDISPATCH}:"*) ;;
    *) export LD_PRELOAD="${GLDISPATCH}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi

if [[ $# -gt 0 && "${1}" != -* ]]; then
  SCENE="$1"
  shift
else
  SCENE="${SCENE:-GdvgFV5R1Z5}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-trajectory_generate/keyboard_trajectories/MP3D/${SCENE}/test}"
CONDA_ENV="${CONDA_ENV:-activemapping}"

echo "[INFO] Habitat/OpenGL environment"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DISPLAY=${DISPLAY}"
echo "  XAUTHORITY=${XAUTHORITY}"
echo "  LD_PRELOAD=${LD_PRELOAD:-}"
echo "  GL_BACKEND=${GL_BACKEND}"
echo "  SCENE=${SCENE}"

python -u trajectory_generate/sim_trajectory.py \
  --dataset MP3D \
  --scene "${SCENE}" \
  --output-dir "${OUTPUT_DIR}" \
  --gl-backend "${GL_BACKEND}" \
  "$@"
