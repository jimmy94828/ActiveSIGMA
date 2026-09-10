### This script is copied from: https://github.com/facebookresearch/Replica-Dataset/blob/main/download.sh
#!/usr/bin/env bash

# Abort on error
set -e

if [ -n "${1:-}" ]; then
  echo -e "\nDownloading and decompressing Replica to $1. The script can resume\npartial downloads -- if your download gets interrupted, simply run it again.\n"
else
  echo "Specify a path to download and decompress Replica to!"
  exit 1
fi

DEST_DIR="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "${DOWNLOAD_DIR}"' EXIT

for p in {a..q}
do
  # Ensure files are continued in case the script gets interrupted halfway through
  wget --continue -P "${DOWNLOAD_DIR}" https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p
done

# Create the destination directory if it doesn't exist yet
mkdir -p "${DEST_DIR}"

cat "${DOWNLOAD_DIR}"/replica_v1_0.tar.gz.part?? | unpigz -p 32 | tar -xvC "${DEST_DIR}"

#download, unzip, and merge the additional habitat configs
wget http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip -P "${DOWNLOAD_DIR}"
unzip -qn "${DOWNLOAD_DIR}/additional_habitat_configs.zip" -d "${DEST_DIR}"