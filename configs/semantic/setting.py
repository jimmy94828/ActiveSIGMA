sem = dict(
    ### Directional semantic voxel map ###
    sem_voxel_nside = 1,  # HEALPix resolution: nside=1 -> 12 bins; 2 -> 48; 4 -> 192.
    sem_voxel_dir_topk = 16,  # Per-direction top-k classes; storage dimension is top-k + residual.
    sem_voxel_alpha_init = 1e-2,  # Positive Dirichlet prior.
    sem_voxel_lambda_h = 0.6,  # Weight of mean directional heat in V(v).
    sem_voxel_lambda_u = 0.4,  # Weight of normalized directional entropy in V(v).
    sem_voxel_heat_s_min = 0.0,  # Minimum directional evidence for JSD evaluation.
    sem_voxel_heat_jsd_use_all_neighbors = True,  # Include unobserved geometric neighbor bins in JSD.

    ### Directional feasibility mask ###
    sem_voxel_feasible_check_depth = 3,  # Probe steps from a voxel toward the candidate-camera side.
    sem_voxel_feasible_beta = 0.75,  # EMA: f_t = beta * f_(t-1) + (1 - beta) * r_t.
    sem_voxel_feasible_tau_low = 0.2,  # f <= tau_low maps to mask value 0.
    sem_voxel_feasible_tau_high = 0.65,  # f >= tau_high maps to mask value 1.
    sem_voxel_feasible_event_driven = True,  # Update only semantic/occupancy dirty voxels.
    sem_voxel_feasible_full_refresh_every = 0,  # 0 disables periodic full refresh.
    sem_voxel_occ_dirty_radius = 1,  # Occupancy-grid neighborhood radius used to mark dirty semantic voxels.

    ### Mapping cadence ###
    sem_voxel_rebuild_freq = 5,  # Rebuild Gaussian-to-voxel indices every N mapping steps.

    ### Optional diagnostics ###
    sem_voxel_value_log_enable = False,  # Print V(v) term breakdowns for selected hot voxels.
    sem_voxel_value_log_every = 1,  # Logging interval in topk_hot_voxels calls.
    sem_voxel_value_log_topk = 5,  # Number of hot voxels shown in each value log.
    sem_voxel_active_dir_log_enable = False,  # Print active-direction coverage statistics.
    sem_voxel_active_dir_log_every = 1,  # Logging interval in topk_hot_voxels calls.
    sem_voxel_active_dir_log_topk = 5,  # Number of hot voxels shown in each direction log.
    flag_debug_vis_eval_voxelheatmap = True,  # Render voxel heat overlays during evaluation.

    ### Mapping keyframe sources ###
    opt_kf_sources = ["local_covisible", "global"],  # Options: local_covisible, global, rskm; None uses legacy replay.
)

planner = dict(
    ### SemanticHeatPlanner NBV score weights ###
    lambda_I = 0.3,  # Directional inconsistency gain (GI).
    lambda_D = 0.25,  # Directional semantic uncertainty gain (GD).
    # lambda_g = 0.3,  # Geometric unexplored-pixel gain.
    lambda_g = 0.0,
    lambda_C = 0.15,  # Travel-distance penalty.

    ### Hot-voxel exploitation candidates ###
    hot_voxel_topk = 50,  # Number of hot voxels considered for exploitation.
    hot_samples_per_voxel = 4,  # Number of highest-heat directions selected per hot voxel.
    hot_sample_radius = 0.5,  # Stage-0 random-radius span in meters.
    hot_voxel_active_only = True,
    hot_voxel_min_value = 0.0,
    hot_voxel_fallback_to_alpha_active = True,
    hot_sample_strict_free_only = True,  # Require strictly free cells; unknown cells are rejected.
    hot_sample_occ_max = -1e-6,  # Occupancy threshold used only when strict_free_only is False.
    hot_sample_clearance_vox = 1,  # Required free-space clearance around a candidate in grid voxels.
    hot_sample_require_free_segment = False,  # Also require the voxel-to-camera segment to be strictly free.
    hot_selection_revalidate = True,  # Recheck the final selected hot candidate.
    hot_sample_radius_min = 0.4,  # Stage-0 random-radius lower bound in meters.

    ### Stage-1 radial sweep ###
    stage1_hot_radial_search = True,  # Sweep fixed radii along each selected hot direction.
    stage1_hot_radius_start = 0.5,  # Inclusive radial-sweep lower bound in meters.
    stage1_hot_radius_end = 1.5,  # Inclusive radial-sweep upper bound in meters.
    stage1_hot_radius_step = 0.25,  # Radial-sweep increment in meters.
    stage1_hot_collect_all_free = True,  # Retain every feasible radius instead of only the first.
    stage1_exploitation_only = False,  # Disable standard exploration candidates during stage 1.
    log_selected_hot_nbv = True,  # Log details when the selected NBV is a hot candidate.

    ### NBV pool pruning and motion guard ###
    nbv_score_thre = 0.0,  # Absolute NBV-score pruning threshold.
    nbv_pool_age_thre = 5,  # Minimum pool age before low-score regular candidates can be removed.
    hot_rm_by_explore_after_age = 5,  # Age after which hot candidates also use geometric pruning.
    hot_nbv_pool_age_thre = 1,  # Minimum pool age before low-score hot candidates can be removed.
    min_nbv_move_dist = 0.0,  # Minimum displacement for regular candidates; 0 disables the guard.
    hot_min_nbv_move_dist = 0.1,  # Minimum displacement for hot candidates.
    hot_key_pos_resolution = 0.0,  # Position quantization; <= 0 uses the exploration voxel size.
    seman_ig_thre = 0.5,  # Semantic-IG threshold used to prune the refinement pool.
)
