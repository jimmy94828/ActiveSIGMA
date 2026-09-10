ROOT=${PWD} 

# Explicitly source conda so it works in non-interactive bash (bash script.sh)
# Adjust this path if anaconda/miniconda is installed elsewhere
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/acm/anaconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"

### create conda environment (skip if already exists) ###
if conda env list | grep -q "^ActiveSIGMA "; then
    echo "[INFO] conda env 'ActiveSIGMA' already exists, skipping create"
else
    conda create -n ActiveSIGMA python=3.8 cmake=3.14.0 -y
fi

### activate conda environment ###
conda activate ActiveSIGMA

# Validate that conda activate actually worked
if [ -z "${CONDA_PREFIX}" ]; then
    echo "[ERROR] conda activate ActiveSIGMA failed - CONDA_PREFIX is empty"
    exit 1
fi

# Use explicit paths so pip/python always target the ActiveSIGMA env,
# even if conda activate doesn't fully switch PATH in this subshell.
# CONDA_PREFIX is set by `conda activate` to the full path of the active env.
ENV_DIR="${CONDA_PREFIX}"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

# Validate python and pip exist in the env
if [ ! -f "${PYTHON}" ] || [ ! -f "${PIP}" ]; then
    echo "[ERROR] python or pip not found in ${ENV_DIR}/bin/"
    exit 1
fi

echo "=== Using Python: $(${PYTHON} --version) ==="
echo "=== Pip target: ${PIP} ==="

# ### Setup habitat-sim ###
cd ${ROOT}/third_parties
# Only clone if the directory does not already exist
[ -d habitat_sim ] || git clone https://github.com/Huangying-Zhan/habitat-sim.git habitat_sim
cd habitat_sim
git submodule update --init --recursive
${PIP} install -r requirements.txt
# Remove stale build dir to avoid 'build/compile_commands.json not found' error
rm -rf build
${PYTHON} setup.py install --headless --bullet

# ### Install thir parties ###
### need to change torch/cuda version to adapt your system cuda version
${PIP} install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f https://download.pytorch.org/whl/cu117/torch_stable.html
${PIP} install git+https://github.com/facebookresearch/pytorch3d.git@05cbea115acbbcbea77999c03d55155b23479991
${PIP} install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
${PIP} install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110

###### install avtivemap
${PIP} install charset_normalizer==2.0.4
${PIP} install mmengine==0.7.3
# These files may not exist in the repo; skip gracefully if missing
${PIP} install -r ${ROOT}/envs/requirements_activemap.txt || echo "[WARN] requirements_activemap.txt not found, skipping"
${PIP} install -r ${ROOT}/envs/requirements_semantic.txt || echo "[WARN] requirements_semantic.txt not found, skipping"

### CoSLAM installation ###
cd ${ROOT}/third_parties/coslam
# Use the current HEAD (main branch) - the original 3bb904e commit doesn't exist in this fork
cd external/NumpyMarchingCubes
${PYTHON} setup.py install

#### install torch_sparse
${PIP} install torch-scatter==2.1.1 torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

## download replica data (scripts may not exist; skip if missing)
# Must cd back to ROOT so wget downloads part files to the right place
cd ${ROOT}
# Remove placeholder text files that the repo ships in data/ (they block mkdir)
for placeholder in data/Replica data/replica_sim_nvs data/mp3d_sim_nvs data/mp3d_data; do
    if [ -f "${ROOT}/${placeholder}" ] && ! [ -d "${ROOT}/${placeholder}" ]; then
        echo "[INFO] Removing placeholder file: ${placeholder}"
        rm -f "${ROOT}/${placeholder}"
    fi
done
bash ${ROOT}/scripts/data/replica_download.sh data/replica_v1 || echo "[WARN] replica_download.sh not found, skipping"
bash ${ROOT}/scripts/data/replica_update.sh data/replica_v1 || echo "[WARN] replica_update.sh not found, skipping"
bash ${ROOT}/scripts/data/replica_slam_download.sh || echo "[WARN] replica_slam_download.sh not found, skipping"

### create soft link if you did not download to the current dir
#ln -s /mnt/Data2/slam_datasets/replica_v1 ./data/replica_v1
#ln -s /mnt/Data2/slam_datasets/Replica ./data/Replica
#ln -s /mnt/Data2/slam_datasets/replica_sim_nvs ./data/replica_sim_nvs
#ln -s /mnt/Data4/slam_datasets/mp3d_sim_nvs ./data/mp3d_sim_nvs
