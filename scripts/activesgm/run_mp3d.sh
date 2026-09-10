#!/bin/bash
##################################################
### This script runs the full NARUTO system
### (active planning and active ray sampling)
### on the MP3D dataset.
##################################################

# Input arguments
scene=${1:-GdvgFV5R1Z5}
num_run=${2:-1}
EXP=${3:-ActiveSem} # can be EXP name or explicit config file path
ENABLE_VIS=${4:-0}
GPU_ID=${5:-0}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-}
export PYTHONNOUSERSITE=1
export PYTHONPATH=
if [ -f /lib/x86_64-linux-gnu/libGLdispatch.so.0 ]; then
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
fi

# Resolve project root from script location so the command works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJ_DIR}"

DATASET=MP3D
RESULT_DIR=${RESULT_ROOT:-${PROJ_DIR}/results}

##################################################
### Random Seed
###     also used to initialize agent pose 
###     from indexing the pose in MP3D SLAM
###     trajectory.
##################################################
seeds=(0 500 1000 1500 1999)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
# scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
scenes=( GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG)          #    YmJkqBEsHnH 
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        if [[ "${EXP}" == /*.py ]] || [[ "${EXP}" == ./*.py ]] || [[ "${EXP}" == ../*.py ]] || [[ "${EXP}" == */*.py ]]; then
            CFG=${EXP}
            EXP_NAME=$(basename "${CFG}" .py)
        else
            CFG=configs/${DATASET}/${scene}/${EXP}.py
            EXP_NAME=${EXP}
        fi

        if [ ! -f "${CFG}" ]; then
            echo "[ERROR] Config file not found: ${CFG}"
            exit 1
        fi

        TRAJ_BASE="data/mp3d_sim_nvs"
        if [ -f "data/mp3d_sim_nvs_v2/${scene}/traj.txt" ]; then
            TRAJ_BASE="data/mp3d_sim_nvs_v2"
        fi

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP_NAME}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        python src/main/activesgm.py --cfg ${CFG} --setting configs/semantic/setting.py --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS}
        run_status=$?
        if [ ${run_status} -ne 0 ]; then
            echo "[ERROR] ActiveSIGMA run failed for scene=${scene}, run=${i}. Skip evaluation."
            exit ${run_status}
        fi

        ### 3D Reconstruction evaluation ###
        GT_MESH=$PROJ_DIR/data/MP3D/v1/scans/${scene}/mesh.obj
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP_NAME}/run_${i}

        python src/evaluation/eval_splatam_recon_v2.py \
        --ckpt ${result_dir}/splatam/exploration_stage_0/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj ${TRAJ_BASE}/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/exploration_stage_0

        python src/evaluation/eval_splatam_recon_v2.py \
        --ckpt ${result_dir}/splatam/exploration_stage_1/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj ${TRAJ_BASE}/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/exploration_stage_1

        python src/evaluation/eval_splatam_recon_v2.py \
        --ckpt ${result_dir}/splatam/final/params.npz \
        --gt_mesh ${GT_MESH} \
        --transform_traj ${TRAJ_BASE}/${scene}/traj.txt \
        --result_dir ${result_dir}/eval_3d/final

    done
done
