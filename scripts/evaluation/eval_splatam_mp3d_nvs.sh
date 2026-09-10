#!/bin/bash
##################################################
### Evaluate SplaTAM/SemSplaTAM-style runs on the
### shared MP3D novel-view trajectory used for
### fair comparison with NARUTO.
##################################################

scene=${1:-gZ6f7yhEvPG}
trial=${2:-0}
EXP=${3:-SemanticHeat}
stage=${4:-final}
GPU_ID=${5:-0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_ID}
PROJ_DIR=${PWD}
DATASET=MP3D
RESULT_DIR=${PROJ_DIR}/results
NVS_EVAL_DATA_BASEDIR=${NVS_EVAL_DATA_BASEDIR:-data/mp3d_sim_nvs_v2}
NVS_EVAL_SUFFIX=${NVS_EVAL_SUFFIX:-$stage}

trials=(0 1 2 3 4)
if [ "$trial" == "all" ]; then
    selected_trials=${trials[@]}
else
    selected_trials=($trial)
fi

scenes=(GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG)
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]}
else
    selected_scenes=($scene)
fi

for scene in $selected_scenes
do
    for i in $selected_trials; do
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        result_dir=${RESULT_DIR}/${DATASET}/${scene}/${EXP}/run_${i}
        eval_data_dir=${NVS_EVAL_DATA_BASEDIR}/${scene}

        echo "==> Evaluating SplaTAM NVS [${result_dir}]"
        if [ ! -f "$CFG" ]; then
            echo "[SKIP] Missing config: ${CFG}"
            continue
        fi
        if [ ! -d "$eval_data_dir" ]; then
            echo "[SKIP] Missing shared NVS eval data: ${eval_data_dir}"
            continue
        fi

        python src/evaluation/eval_nvs_result.py \
            --cfg $CFG \
            --result_dir $result_dir \
            --stage $stage \
            --eval_data_dir $eval_data_dir \
            --eval_suffix $NVS_EVAL_SUFFIX
    done
done
