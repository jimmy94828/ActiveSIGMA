import numpy as np
import os

_base_ = "../../default.py"

general = dict(
    dataset="Replica",
    scene="office3",
    num_iter=2000,
    device="cuda",
)

dirs = dict(
    data_dir="data/",
    result_dir="results/",
    cfg_dir=os.path.join("configs", general["dataset"], general["scene"]),
)

sim = dict(method="habitat_v2")
if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg=os.path.join(dirs["cfg_dir"], "habitat.py")
    )

slam = dict(method="splatam")

if slam["method"] == "splatam":
    slam.update(
        room_cfg=os.path.join("configs", general["dataset"], "replica_splatam_s.py"),
        enable_active_planning=False,
        use_global_keyframe=False,
        override=dict(
            map_every=5,
            report_global_progress_every=5,
            tracking=dict(
                use_gt_poses=False,
            ),
        ),
    )

planner = dict(
    method="predefined_traj",
    up_dir=np.array([0, 0, 1]),
    use_traj_pose=True,
    SLAMData_dir=os.path.join(dirs["data_dir"], "Replica", general["scene"]),
)

visualizer = dict(
    method="active_lang",
    vis_rgbd=True,
    vis_rgbd_max_depth=10,
)
