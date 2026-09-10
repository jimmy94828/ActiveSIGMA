#!/usr/bin/env bash
set -euo pipefail

# Habitat-Sim rendering follows the same OpenGL/X setup used by the ActiveSGM
# main scripts. Override any of these from the shell when needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,0}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
export GL_BACKEND="${GL_BACKEND:-x11}"


GLDISPATCH=/lib/x86_64-linux-gnu/libGLdispatch.so.0
if [[ -f "${GLDISPATCH}" ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":${GLDISPATCH}:"*) ;;
    *) export LD_PRELOAD="${GLDISPATCH}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
  esac
fi

echo "[INFO] Habitat/OpenGL environment"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  DISPLAY=${DISPLAY}"
echo "  XAUTHORITY=${XAUTHORITY}"
echo "  LD_PRELOAD=${LD_PRELOAD:-}"
echo "  GL_BACKEND=${GL_BACKEND}"

if [[ $# -gt 0 && "${1}" != -* ]]; then
  SCENE="$1"
  shift
else
  SCENE="${SCENE:-office0}"
fi
echo "  SCENE=${SCENE}"

python -u trajectory_generate/sim_trajectory.py \
  --dataset Replica \
  --scene "${SCENE}" \
  --gl-backend "${GL_BACKEND}" \
  "$@"
