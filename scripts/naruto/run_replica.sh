#!/bin/bash
##################################################
### This script is to run the full NARUTO system
### (active planning and active ray sampling)
###  on the Replica dataset.
##################################################

# Input arguments
scene=${1:-office0}
num_run=${2:-1}
EXP=${3:-NARUTO} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/home/phudh/.Xauthority}
export LD_PRELOAD=${LD_PRELOAD:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}
PROJ_DIR=${PWD}
DATASET=Replica
RESULT_DIR=${PROJ_DIR}/results
RUN_EVAL=${NARUTO_RUN_EVAL:-1}

##################################################
### Random Seed
###     also used to initialize agent pose
###     from indexing the pose in Replica SLAM
###     trajectory.
##################################################
seeds=(0 500 1000 1500 1999)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
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

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        if [ ! -f "$CFG" ]; then
            echo "[SKIP] Missing config: ${CFG}"
            continue
        fi

        if python src/main/naruto.py --cfg ${CFG} --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS}; then
            if [ "$RUN_EVAL" = "1" ]; then
                bash scripts/evaluation/eval_replica.sh ${scene} ${i} 2000 ${EXP}
            fi
        else
            status=$?
            echo "[ERROR] NARUTO failed for ${DATASET}/${scene}/${EXP}/run_${i} with exit code ${status}. Skip evaluation for this run."
        fi
    done
done
