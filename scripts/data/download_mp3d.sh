#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MP3D_ROOT=${1:-${PROJ_DIR}/data/MP3D}

mkdir -p "${MP3D_ROOT}"
cd "${PROJ_DIR}"
while read p; do
  python ./src/data/download_mp.py -o "${MP3D_ROOT}" --id "$p" --task_data habitat
done <"${SCRIPT_DIR}/scan_id.txt"
