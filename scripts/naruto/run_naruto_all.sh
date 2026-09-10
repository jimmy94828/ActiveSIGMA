#!/bin/bash



# Input arguments

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export DISPLAY=${DISPLAY:-:1}
export XAUTHORITY=${XAUTHORITY:-/home/phudh/.Xauthority}
export LD_PRELOAD=${LD_PRELOAD:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}
PROJ_DIR=${PWD}
DATASET=Replica
RESULT_DIR=/media/phudh/HDD/ActiveMapping_ablation/results
RUN_EVAL=${NARUTO_RUN_EVAL:-1}
scenes=(room0 room1 room2 office0 office1 office2 office3 office4)

NARUTO_RENDER_EVAL_EVERY=${NARUTO_RENDER_EVAL_EVERY:-5}
NARUTO_RENDER_CHUNK_SIZE=${NARUTO_RENDER_CHUNK_SIZE:-8192}
NARUTO_RENDER_LPIPS_DOWNSAMPLE=${NARUTO_RENDER_LPIPS_DOWNSAMPLE:-1}
NARUTO_EVAL_DATA_BASEDIR=${NARUTO_EVAL_DATA_BASEDIR:-data/replica_sim_nvs}
NARUTO_EVAL_POSE_MODE=${NARUTO_EVAL_POSE_MODE:-absolute}
append_render_args() {
    RENDER_ARGS=()
    RENDER_DATA_ARGS=()
    if [ "${NARUTO_RENDER_SAVE_FRAMES:-0}" = "1" ]; then
        RENDER_ARGS+=(--save_frames)
    fi
    if [ "${NARUTO_RENDER_DISABLE_LPIPS:-0}" = "1" ]; then
        RENDER_ARGS+=(--disable_lpips)
    fi
    if [ "${NARUTO_RENDER_MAX_FRAMES:-0}" != "0" ]; then
        RENDER_ARGS+=(--max_frames ${NARUTO_RENDER_MAX_FRAMES})
    fi

    if [ -n "${NARUTO_EVAL_DATA_DIR:-}" ]; then
        eval_data_dir=${NARUTO_EVAL_DATA_DIR}
    else
        eval_data_dir=${NARUTO_EVAL_DATA_BASEDIR}/${scene}
    fi
    if [ -d "$eval_data_dir" ]; then
        RENDER_DATA_ARGS+=(--eval_data_dir $eval_data_dir --eval_pose_mode $NARUTO_EVAL_POSE_MODE)
    else
        echo "[WARN] Shared eval data not found: ${eval_data_dir}; render eval will fall back to active trajectory."
    fi
}


for scene in "${scenes[@]}"; do
    result_dir=${RESULT_DIR}/${DATASET}/$scene/NARUTO/run_0
    result_txt=${result_dir}/eval_result.txt
    mkdir -p ${result_dir}
    : > $result_txt
    echo "==> Evaluating [${result_dir}]"

    CKPT=$result_dir/coslam/checkpoint/ckpt_2000_final.pt
    INPUT_MESH=$result_dir/coslam/mesh/mesh_2000_final.ply
    REC_MESH=$result_dir/coslam/mesh/mesh_2000_final_cull_occlusion.ply
    scene_prefix=${scene%?}
    scene_id=${scene: -1}
    DASHSCENE=${scene_prefix}_${scene_id}
    GT_MESH=$PROJ_DIR/data/replica_v1/${DASHSCENE}/mesh.ply
    CFG=configs/${DATASET}/${scene}/NARUTO.py
    COSLAM_CFG=configs/${DATASET}/${scene}/coslam.yaml
    RENDER_EVAL_DIR=$result_dir/coslam/eval_final

    python src/evaluation/eval_coslam_render.py \
        --cfg $CFG \
        --result_dir $result_dir \
        --ckpt $CKPT \
        --eval_dir $RENDER_EVAL_DIR \
        --result_txt $result_txt \
        --eval_every $NARUTO_RENDER_EVAL_EVERY \
        --chunk_size $NARUTO_RENDER_CHUNK_SIZE \
        --lpips_downsample $NARUTO_RENDER_LPIPS_DOWNSAMPLE \
        --skip_first \
        "${RENDER_DATA_ARGS[@]}" \
        "${RENDER_ARGS[@]}"
        REC_MESH_FOR_EVAL=$INPUT_MESH
        if [ -f "$INPUT_MESH" ]; then
            echo "==> Culling mesh"
            if python third_parties/neural_slam_eval/cull_mesh.py \
                --config $COSLAM_CFG \
                --input_mesh $INPUT_MESH \
                --ckpt_path $CKPT \
                --remove_occlusion; then
                if [ -f "$REC_MESH" ]; then
                    REC_MESH_FOR_EVAL=$REC_MESH
                fi
            else
                echo "[WARN] Mesh culling failed; falling back to uncropped mesh: ${INPUT_MESH}"
            fi
        else
            echo "[WARN] Missing input mesh; skip mesh reconstruction metrics: ${INPUT_MESH}"
        fi

        if [ -f "$REC_MESH_FOR_EVAL" ] && [ -f "$GT_MESH" ]; then
            echo "==> Evaluating reconstruction result [accuracy, completeness, and completion ratio]"
            python src/evaluation/eval_recon.py \
                --rec_mesh $REC_MESH_FOR_EVAL \
                --gt_mesh $GT_MESH \
                --result_txt $result_txt
        else
            echo "[WARN] Skip reconstruction metrics; missing rec mesh or GT mesh."
        fi

        if [ -f "$GT_MESH" ]; then
            echo "==> Evaluating reconstruction result [MAD]"
            python src/evaluation/eval_mad.py \
                --cfg $CFG \
                --ckpt $CKPT --gt_mesh $GT_MESH \
                --result_txt $result_txt
        else
            echo "[WARN] Skip MAD; missing GT mesh: ${GT_MESH}"
        fi

        echo "==> Evaluating trajectory length (m)"
        python src/evaluation/eval_traj_length.py \
            --ckpt ${CKPT} \
            --result_txt $result_txt

        echo "==> Evaluating render result [PSNR, MS-SSIM, LPIPS, depth L1/RMSE]"
        append_render_args
        python src/evaluation/eval_coslam_render.py \
            --cfg $CFG \
            --result_dir $result_dir \
            --ckpt $CKPT \
            --eval_dir $RENDER_EVAL_DIR \
            --result_txt $result_txt \
            --eval_every $NARUTO_RENDER_EVAL_EVERY \
            --chunk_size $NARUTO_RENDER_CHUNK_SIZE \
            --lpips_downsample $NARUTO_RENDER_LPIPS_DOWNSAMPLE \
            --skip_first \
            "${RENDER_DATA_ARGS[@]}" \
            "${RENDER_ARGS[@]}"
    done