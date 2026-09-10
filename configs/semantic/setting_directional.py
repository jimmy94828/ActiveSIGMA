sem = dict(
    ### SemanticVoxelMap 方向性Dirichlet和HEALPix ###
    sem_voxel_nside        = 1,                 # HEALPix nside（12*nside^2 個區塊；nside=1→1 2→4 8→192）
    sem_voxel_eta          = 0.95,              # 遺忘係數
    sem_voxel_w_i          = 0.6,               # JSD 熱度權重 not used now
    sem_voxel_w_n          = 0.4,               # 觀測稀疏懲罰 not used now
    sem_voxel_dir_topk     = 16,                # directional 儲存 top-k 類別（實際維度為 k+1，含 residual）
    sem_voxel_alpha_init   = 1e-2,              # Dirichlet 初始先驗
    sem_voxel_lambda_h     = 0.6,               # V(v) 中 Hbar 權重  origin 0.6
    sem_voxel_lambda_u     = 0.4,               # V(v) 中 Ehat 權重  origin 0.4
    GD_threshold           = 0.0,               # GD 僅加總 value > GD_threshold 的 voxel ##################################
    sem_voxel_heat_s_min   = 0.0,               # 鄰居 JSD 計算的最小證據量  origin None
    sem_voxel_heat_jsd_use_all_neighbors = True, # True: H_v[p] 與所有幾何鄰居算 JSD，不要求鄰居先 observed

    ### feasibility mask 更新參數 ###
    sem_voxel_feasible_check_depth = 3,         # 沿 -d 探查步數
    sem_voxel_feasible_gamma       = 0.8,       # 舊版參數（保留相容）
    sem_voxel_feasible_beta        = 0.75,       # f_t = beta*f_{t-1} + (1-beta)*r_t
    sem_voxel_feasible_tau_low     = 0.2,       # f <= tau_low -> m=0 origin 0.3
    sem_voxel_feasible_tau_high    = 0.65,       # f >= tau_high -> m=1 origin 0.7
    sem_voxel_feasible_event_driven = True,
    sem_voxel_feasible_full_refresh_every = 5, # 20
    sem_voxel_occ_dirty_radius = 1,

    ### 映射更新頻率 ###
    sem_voxel_rebuild_freq = 5,                # Gaussian to voxel 索引重建頻率（step 數）

    ### debug log: voxel value V(v) = lambda_h*Hbar + lambda_u*Ehat ###
    sem_voxel_value_log_enable = True,          # True: 列印 voxel value 的兩個 term
    sem_voxel_value_log_every  = 1,             # 每幾次 topk_hot_voxels 呼叫印一次
    sem_voxel_value_log_topk   = 5,             # 每次最多列印幾個 top voxel
    sem_voxel_active_dir_log_enable = True,     # True: 列印 active direction 數量統計
    sem_voxel_active_dir_log_every  = 1,        # 每幾次 topk_hot_voxels 呼叫印一次
    sem_voxel_active_dir_log_topk   = 5,        # 每次最多列印幾個 top voxel 的 active dir 數

    ### pixel-level weighted directional evidence ###
    sem_voxel_pixel_alpha = 1.0,                # omega: observation confidence exponent
    sem_voxel_pixel_beta = 1.0,                 # omega: agreement exponent
    sem_voxel_pixel_gamma = 1.0,                # reliability: silhouette exponent
    sem_voxel_pixel_delta = 1.0,                # reliability: render confidence exponent
    sem_voxel_pixel_w_min = 0.5,                # omega lower-bound blend weight under disagreement
    sem_voxel_pixel_reliability_use_render_entropy = True,
    sem_voxel_pixel_conflict_enable = True,     # maintain z_{v,p} conflict score for exploration signal
    sem_voxel_pixel_weight_log_enable = True,
    sem_voxel_pixel_weight_log_every = 20,

    ### debug visualization ###
    flag_debug_vis_eval_voxelheatmap = True,    # True: 在 eval 輸出中額外繪製 voxel heat value 視覺化
)
planner = dict(
    ### SemanticHeatPlanner NBV score weights ###
    lambda_I              = 0.3,                # GI：方向熱度聚合
    lambda_D              = 0.25,               # GD：語意不確定
    lambda_g              = 0.3,                # G_g：幾何缺口（explore_ig）
    lambda_C              = 0.15,               # Cost：移動距離懲罰
    hot_voxel_topk        = 50,                 # legacy voxel budget fallback
    hot_samples_per_voxel = 4,                  # legacy per-voxel direction cap fallback
    hot_direction_topk    = 200,                # global top-k (voxel, direction) candidates
    hot_direction_max_per_voxel = 4,            # global ranking 前每個 voxel 最多保留幾個方向
    hot_direction_score_w_dir = 1.0,            # score(v,p) 中 H_v[p] 權重
    hot_direction_score_w_voxel = 1.0,          # score(v,p) 中 voxel value V(v) 權重
    hot_sample_radius     = 0.5,                # 取樣半徑（meter）
    hot_voxel_active_only = True,
    hot_voxel_min_value = 0.0,
    hot_voxel_fallback_to_alpha_active = True,
    hot_sample_strict_free_only = True,         # True: HOT 候選只允許 strict free（unknown 不可）
    hot_sample_occ_max    = -1e-6,              # 候選點可行門檻：strict free-only（occ<0）
    hot_sample_clearance_vox = 0,               # HOT 候選周圍安全距離（體素）；1 可減少貼障礙候選
    hot_sample_require_free_segment = False,    # True: 強制 voxel->camera 全路徑都在 free space
    hot_selection_revalidate = True,            # True: 最終選中 HOT 候選前再做一次安全檢查
    hot_sample_radius_min = 0.4,                # 取樣半徑下界 (meter) not used now
    hot_voxel_max_samples = 4,                  # 每個 hot voxel 最大採樣數 (0=不限制)
    hot_voxel_limit_reset_per_stage = False,    # True: 每個 exploration stage 重新計數

    # hot voxel sampling part
    stage1_hot_radial_search = True,            # stage 1: 沿 hot direction 固定半徑掃描
    stage1_hot_radius_start = 0.5,              # 掃描半徑起點（meter）
    stage1_hot_radius_end   = 1.5,              # 掃描半徑終點（meter）
    stage1_hot_radius_step  = 0.25,             # 掃描半徑步長（meter）
    stage1_hot_collect_all_free = True,         # True: 將半徑範圍內所有可行 free 點都納入候選
    stage1_exploitation_only = False,           # True: stage 1 僅使用 hot voxel exploitation 候選
    log_selected_hot_nbv = True,                # True: 當最終 NBV 來自 hot voxel 候選時印出詳細資訊

    ### NBV pool pruning ###
    nbv_score_thre        = 0.03,               # NBV 絕對門檻
    nbv_pool_age_thre     = 5,                  # 候選在 pool 的評分輪數需 > 此門檻才可被低分刪除
    hot_rm_by_explore_after_age = 3,            # HOT 候選前幾輪僅以 revisit 刪除，之後也受 explore_thre 幾何刪除
    hot_nbv_pool_age_thre = 1,                  # HOT 候選低分刪除的最小 pool_age（通常小於一般候選）
    min_nbv_move_dist = 0.0,                    # 一般候選最小位移門檻（meter）；0 表示不啟用
    hot_min_nbv_move_dist = 0.2,                # HOT 候選最小位移門檻（meter），避免重複選到原地   origin 0.05
    hot_key_pos_resolution = 0.0,               # HOT key 位置量化解析度（meter）；<=0 代表使用 voxel_size

    ### NBV score normalization ### not used now
    score_norm_low_pct    = 10.0,               # robust normalization 下界百分位
    score_norm_high_pct   = 90.0,               # robust normalization 上界百分位

    ### scoring switch ###
    use_semantic_gains    = True,               # True: 使用 GI/GD+geometry+cost

    seman_ig_thre         = 0.5,                # semantic IG threshold for refinement pool pruning
)
