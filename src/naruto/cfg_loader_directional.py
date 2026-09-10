"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import mmengine

def override_cfg(
        args: argparse.Namespace,
        cfg : mmengine.Config
    ) -> mmengine.Config:
    """override configuration

    Args:
        args: arguments
        cfg : configuration

    Returns:
        cfg : updated configuration
    """
    def _copy_if_exists(dst_cfg, src_cfg, key: str) -> None:
        if hasattr(src_cfg, key):
            setattr(dst_cfg, key, getattr(src_cfg, key))

    # override configuration with arguments
    if hasattr(args, "setting") and args.setting is not None:       # setting for semantic voxelization and planner
        sem_keys = [
            "sem_voxel_nside",
            "sem_voxel_dir_topk",
            "sem_voxel_alpha_init",
            "sem_voxel_lambda_h",
            "sem_voxel_lambda_u",
            "sem_voxel_heat_s_min",
            "sem_voxel_heat_jsd_use_all_neighbors",
            "sem_voxel_feasible_check_depth",
            "sem_voxel_feasible_beta",
            "sem_voxel_feasible_tau_low",
            "sem_voxel_feasible_tau_high",
            "sem_voxel_feasible_event_driven",
            "sem_voxel_feasible_full_refresh_every",
            "sem_voxel_occ_dirty_radius",
            "sem_voxel_rebuild_freq",
            "sem_voxel_value_log_enable",
            "sem_voxel_value_log_every",
            "sem_voxel_value_log_topk",
            "sem_voxel_active_dir_log_enable",
            "sem_voxel_active_dir_log_every",
            "sem_voxel_active_dir_log_topk",
            "flag_debug_vis_eval_voxelheatmap",
            "opt_kf_sources",
        ]
        for k in sem_keys:
            _copy_if_exists(cfg.slam, args.setting.sem, k)
    
    if hasattr(args, "setting") and args.setting is not None:       # setting for semantic voxelization and planner
        planner_keys = [
            "lambda_I",
            "lambda_D",
            "lambda_g",
            "lambda_C",
            "hot_voxel_topk",
            "hot_samples_per_voxel",
            "hot_direction_topk",
            "hot_direction_max_per_voxel",
            "hot_direction_score_w_dir",
            "hot_direction_score_w_voxel",
            "hot_sample_radius",
            "hot_voxel_active_only",
            "hot_voxel_min_value",
            "hot_voxel_fallback_to_alpha_active",
            "hot_sample_strict_free_only",
            "hot_sample_occ_max",
            "hot_sample_clearance_vox",
            "hot_sample_require_free_segment",
            "hot_selection_revalidate",
            "hot_sample_radius_min",
            "stage1_hot_radial_search",
            "stage1_hot_radius_start",
            "stage1_hot_radius_end",
            "stage1_hot_radius_step",
            "stage1_hot_collect_all_free",
            "stage1_exploitation_only",
            "log_selected_hot_nbv",
            "nbv_score_thre",
            "nbv_pool_age_thre",
            "hot_rm_by_explore_after_age",
            "hot_nbv_pool_age_thre",
            "min_nbv_move_dist",
            "hot_min_nbv_move_dist",
            "hot_key_pos_resolution",
            "seman_ig_thre",
        ]
        for k in planner_keys:
            _copy_if_exists(cfg.planner, args.setting.planner, k)

    if hasattr(args, "seed") and args.seed is not None:
        ### random seed ###
        cfg.general.seed = args.seed

    if hasattr(args, "result_dir") and args.result_dir is not None:
        ### output/result directory ###
        cfg.dirs.result_dir = args.result_dir

    if hasattr(args, "enable_vis") and args.enable_vis is not None:
        ### output/result directory ###
        enable_vis = args.enable_vis == 1
        cfg.visualizer.vis_rgbd = enable_vis
    return cfg


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to run NARUTO."
        )
    parser.add_argument("--cfg", type=str, default="configs/default.py",
                        help="NARUTO config")
    parser.add_argument("--setting", type=str, default="configs/semantic/setting.py",
                        help="Semantic voxelize config (sem + planner overrides)")
    parser.add_argument("--result_dir", type=str, default=None, 
                        help="result directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; also used as the initial pose idx for Replica")
    parser.add_argument("--enable_vis", type=int, default=None,
                        help="enable visualization. 1: True, 0: False")
    parser.add_argument("--stage", type=str, default='final',
                        help="ONLY for SplaTAM result evaluation ")
    args = parser.parse_args()
    return args


def load_cfg(args: argparse.Namespace) -> mmengine.Config:
    """argument parsing and load configuration

    Args:
        args: arguments

    Returns:
        cfg : configuration

    """
    cfg = mmengine.Config.fromfile(args.cfg)
    # load semantic setting file into args.setting so override_cfg can read it
    if hasattr(args, "setting") and args.setting is not None:
        args.setting = mmengine.Config.fromfile(args.setting)
    cfg = override_cfg(args, cfg)
    return cfg
