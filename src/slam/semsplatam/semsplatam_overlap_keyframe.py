"""
We have reused part of SplaTAM's code in this file.
For SplaTAM License, refer to https://github.com/spla-tam/SplaTAM/blob/main/LICENSE.
"""
import os.path
import sys
import mmengine
import numpy as np
from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple
from PIL import Image
import json
import torch
import torch.nn.functional as F

from src.slam.splatam.splatam import SplatamOurs
from src.slam.splatam.eval_helper import eval, report_progress
from src.utils.general_utils import InfoPrinter
from src.slam.splatam.exploration_map import ExplorationMap
from third_parties.splatam.utils.slam_external import calc_psnr,calc_ssim

### original Splatam modules ###
sys.path.append("third_parties/splatam")
from scripts.splatam import get_dataset, initialize_camera_pose
from utils.slam_helpers import (
    matrix_to_quaternion, transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette
)
from datasets.gradslam_datasets import (load_dataset_config,)
from utils.keyframe_selection import keyframe_selection_overlap
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians
from utils.eval_helpers import report_loss#, report_progress
from utils.common_utils import save_params

from transformers import AutoProcessor, AutoModelForUniversalSegmentation


# modified version
from src.slam.splatam.modified_ver.scripts.splatam import *
from src.slam.semsplatam.modified_ver.splatam.eval_helper import *
from src.slam.semsplatam.modified_ver.semantic.oneformer import oneformer_segmentation,positive_normalize
from src.slam.semsplatam.modified_ver.splatam.splatam import *
from src.slam.semsplatam.modified_ver.splatam.export_helper import *
from src.data.finetune_oneformer_ReplicaV2 import modify_metadata
from src.slam.semsplatam.modified_ver.scripts.splatam import get_dataset
PRINT_INFO = True

# modified semantic voxel map
from src.slam.semsplatam.semantic_voxel_map import SemanticVoxelMap         # including the directional heat and entropy query functions
from src.utils.healpix_utils import dir_to_bin                              # for mapping 3D directions to HEALPix bins

class SemSplatam(SplatamOurs):
    def __init__(self,
                 main_cfg: mmengine.Config, info_printer: InfoPrinter, logger: SummaryWriter
                 ) -> None:
        '''
        refer to: https://github.com/KBH00/Semantic-Fast-SAM/blob/gpu_ver/main_ssa_engine.py
        https://github.com/oppo-us-research/ActiveLang/blob/lychen/dev/src/slam/semsplatam/semsplatam.py
        https://github.com/ArthurBrussee/brush
        https://github.com/ShuhongLL/SGS-SLAM/tree/main
       '''
        SplatamOurs.__init__(self, main_cfg, info_printer, logger)
        ### load eval dataset with semantic
        self.load_eval_dataset_with_semantic()
        ### loading in segmantation network and Langeuage encoder ###
        self.semantic_device = self.slam_cfg['semantic_device']
        semantic_device_text = str(self.semantic_device)
        if semantic_device_text.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"[warning] semantic_device={semantic_device_text} requested but CUDA is unavailable; using cpu")
                self.semantic_device = "cpu"
            else:
                try:
                    semantic_device_idx = int(semantic_device_text.split(":", 1)[1]) if ":" in semantic_device_text else torch.cuda.current_device()
                except (TypeError, ValueError):
                    semantic_device_idx = 0
                if semantic_device_idx >= torch.cuda.device_count():
                    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
                    print(
                        f"[warning] semantic_device={semantic_device_text} is unavailable "
                        f"with CUDA_VISIBLE_DEVICES={visible_devices}; using cuda:0"
                    )
                    self.semantic_device = "cuda:0"
        self.oneformer_processor = AutoProcessor.from_pretrained(self.slam_cfg['ade20k_checkpoint'])
        self.oneformer_model = AutoModelForUniversalSegmentation.from_pretrained(
            self.slam_cfg['oneformer_checkpoint'],is_training=False).to(self.semantic_device)
        self.n_cls = self.slam_cfg['num_semantic_classes']
        self.topk = self.slam_cfg['num_topk_logits']
        self.oneformer_processor.image_processor.num_text = self.oneformer_model.config.num_queries - self.oneformer_model.config.text_encoder_n_ctx

        class_info_file = self.slam_cfg['class_info_file']
        if os.path.exists(class_info_file):
            self.class_info_file = class_info_file
        else:
            semantic_dir = self.slam_cfg.get('semantic_dir', '')
            info_semantic_file = os.path.join(semantic_dir, 'info_semantic.json')
            if os.path.exists(info_semantic_file):
                self.init_replica_config_cls2label(info_semantic_file, class_info_file)
                self.class_info_file = class_info_file
            else:
                raise FileNotFoundError(
                    "Missing semantic class metadata. Expected either existing "
                    f"class_info_file at '{class_info_file}' or fallback file "
                    f"'{info_semantic_file}'."
                )

        modify_metadata(class_info_file=self.class_info_file, processor=self.oneformer_processor)
        # Keep semantic voxel-map config so the bbox can be re-derived from sim2slam later.
        self._semantic_voxel_kwargs = dict(
            voxel_size = self.slam_cfg['bbox_voxel_size'],
            n_classes  = int(self.n_cls),
            nside      = int(getattr(self.slam_cfg, 'sem_voxel_nside', 1)),
            alpha_init = float(getattr(self.slam_cfg, 'sem_voxel_alpha_init', 1e-2)),
            lambda_h   = float(getattr(self.slam_cfg, 'sem_voxel_lambda_h', 0.6)),
            lambda_u   = float(getattr(self.slam_cfg, 'sem_voxel_lambda_u', 0.4)),
            heat_s_min = getattr(self.slam_cfg, 'sem_voxel_heat_s_min', None),
            feasible_beta = float(getattr(self.slam_cfg, 'sem_voxel_feasible_beta', 0.85)),
            feasible_tau_low = float(getattr(self.slam_cfg, 'sem_voxel_feasible_tau_low', 0.3)),
            feasible_tau_high = float(getattr(self.slam_cfg, 'sem_voxel_feasible_tau_high', 0.7)),
            value_log_enable = bool(getattr(self.slam_cfg, 'sem_voxel_value_log_enable', False)),
            value_log_every = int(getattr(self.slam_cfg, 'sem_voxel_value_log_every', 20)),
            value_log_topk = int(getattr(self.slam_cfg, 'sem_voxel_value_log_topk', 5)),
            active_dir_log_enable = bool(getattr(self.slam_cfg, 'sem_voxel_active_dir_log_enable', False)),
            active_dir_log_every = int(getattr(self.slam_cfg, 'sem_voxel_active_dir_log_every', 20)),
            active_dir_log_topk = int(getattr(self.slam_cfg, 'sem_voxel_active_dir_log_topk', 5)),
            heat_jsd_use_all_neighbors = bool(getattr(self.slam_cfg, 'sem_voxel_heat_jsd_use_all_neighbors', False)),
            device     = self.device,
            topk       = int(getattr(self.slam_cfg, 'sem_voxel_dir_topk', self.topk)),
        )
        self.semantic_voxel_map = self._build_semantic_voxel_map(
            bbox_bound=np.asarray(self.slam_cfg['bbox_bound'], dtype=np.float32)
        )
        # Gaussian to voxel index mapping (for fast GI/GD evaluation)
        self.gaussian_voxel_idx = None        # Tensor[N_gauss] of voxel indices (long), -1 for OOB
        # Voxel to Gaussian list mapping (for fast GI/GD evaluation)
        self.voxel_gaussian_lists = {}        # dict[v_idx(int) -> List[gauss_idx(int)]]

    def _build_semantic_voxel_map(self, bbox_bound: np.ndarray) -> SemanticVoxelMap:
        """Create a SemanticVoxelMap with the current semantic configuration."""
        return SemanticVoxelMap(
            bbox_bound=np.asarray(bbox_bound, dtype=np.float32),
            **self._semantic_voxel_kwargs,
        )

    def _transform_bbox_sim_to_slam(self, bbox_bound: np.ndarray, sim2slam: torch.Tensor) -> np.ndarray:
        """Transform a simulator-world AABB into the corresponding SLAM-world AABB."""
        bbox_bound = np.asarray(bbox_bound, dtype=np.float32)
        mins = bbox_bound[:, 0]
        maxs = bbox_bound[:, 1]
        corners = np.array(
            [[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])],
            dtype=np.float32,
        )
        corners_h = torch.cat(
            [
                torch.from_numpy(corners).to(device=sim2slam.device, dtype=torch.float32),
                torch.ones((corners.shape[0], 1), device=sim2slam.device, dtype=torch.float32),
            ],
            dim=1,
        )
        corners_slam = (sim2slam @ corners_h.T).T[:, :3]
        mins_slam = corners_slam.min(dim=0).values.detach().cpu().numpy()
        maxs_slam = corners_slam.max(dim=0).values.detach().cpu().numpy()
        return np.stack([mins_slam, maxs_slam], axis=1).astype(np.float32)

    def rebuild_gaussian_voxel_index(self) -> None:
        """
        Rebuild mapping from Gaussians (their centers) to voxel indices.
        - Stores `self.gaussian_voxel_idx` as a LongTensor of length N_gauss with -1 for out-of-bounds.
        - Builds `self.voxel_gaussian_lists` as a python dict mapping voxel_idx -> list of gaussian indices.
        """
        # try several possible keys used across versions
        candidate_keys = ['means3D', 'means3d', 'means3D_cpu', 'means', 'mu', 'centers']
        mu = None
        for k in candidate_keys:                # find the position key used in the current SplaTAM version
            if k in self.params:
                mu = self.params[k]
                break
        if mu is None:
            # nothing to do
            return

        # ensure mu is a CPU/GPU tensor of shape [N,3]
        if isinstance(mu, list):
            mu = torch.tensor(mu, device=self.device, dtype=torch.float32)
        else:
            mu = mu.detach() if hasattr(mu, 'detach') else mu
            mu = mu.to(self.device).float()

        if mu.numel() == 0 or mu.dim() != 2 or mu.size(1) < 3:
            return

        # take first three coords if extra dims
        mu_xyz = mu[:, :3]

        # voxelize gaussian centers to get voxel indices (also get in-bounds mask)
        v_idx, inb = self.semantic_voxel_map.voxelize_points(mu_xyz, return_mask=True)
        # mark out-of-bounds gaussians with -1 and store on CPU for lightweight indexing
        v_idx = v_idx.long()
        if inb is not None:
            inb_mask = inb.to(torch.bool)           # in boundary
            neg_one = torch.full_like(v_idx, -1, dtype=v_idx.dtype)
            v_idx = torch.where(inb_mask, v_idx, neg_one)
        self.gaussian_voxel_idx = v_idx.detach().cpu().long()

        # build the voxel to gaussian list mapping
        self.voxel_gaussian_lists = {}
        v_idx_cpu = self.gaussian_voxel_idx.detach().cpu().long()
        for gauss_i, v in enumerate(v_idx_cpu.tolist()):
            if v < 0:
                continue
            self.voxel_gaussian_lists.setdefault(int(v), []).append(int(gauss_i))

    @torch.no_grad()
    def eval_candidate_semantic_gains(self, c2w: torch.Tensor, seen: torch.Tensor = None) -> Tuple[float, float]:
            """
            Compute GI (direction heat aggregation) and GD (voxel semantic entropy aggregation)
            for a candidate camera pose.

            Args:
                c2w: [4,4] camera-to-world pose (SplaTAM system)
                seen: optional boolean mask over gaussians (same ordering as rendered radii>0)

            Returns:
                (GI, GD) both floats (0.0 when no visible voxels)

            Formulas:
                GI(T) = sum_v w_v * H_v[p_T(v)] / (sum_v w_v + eps)
                GD(T) = sum_v w_v * Ehat_v      / (sum_v w_v + eps)
                # pre: Ehat_v = normalized entropy of voxel v's semantic distribution (normalized to [0,1] by log(num_classes))
                Ehat_v = normalized mean entropy over active direction bins of voxel v
                (i.e. per-voxel mean of directional entropies, already scaled to [0,1])
            """
            if self.gaussian_voxel_idx is None:
                return 0.0, 0.0

            # seen should be a 1-D boolean mask over gaussians
            if seen is None:
                return 0.0, 0.0

            # ensure seen is a tensor
            if not isinstance(seen, torch.Tensor):
                seen = torch.tensor(seen)

            # gaussian indices are stored on CPU for lightweight indexing
            gauss_idx = self.gaussian_voxel_idx
            if not isinstance(gauss_idx, torch.Tensor):
                gauss_idx = torch.tensor(gauss_idx)

            gauss_dev = gauss_idx.device

            # convert seen mask to bool tensor on the same device as gauss_idx
            # Ensure the mask is flattened to 1-D so indexing/trimming works reliably
            seen_bool = seen.to(gauss_dev).bool().view(-1)

            # if lengths mismatch, trim or pad the seen mask to match current gaussian count
            n_gauss = int(gauss_idx.numel())
            if seen_bool.numel() > n_gauss:
                # trim extra entries (render outputs may be longer than current gaussian list)
                seen_bool = seen_bool[:n_gauss]
            elif seen_bool.numel() < n_gauss:
                # pad with False if shorter
                pad = torch.zeros((n_gauss - seen_bool.numel(),), dtype=torch.bool, device=gauss_dev)
                seen_bool = torch.cat([seen_bool, pad], dim=0)

            seen_idx = torch.where(seen_bool)[0]
            if seen_idx.numel() == 0:
                return 0.0, 0.0

            # gather visible gaussian -> voxel indices (safe indexing)
            try:
                visible_v = torch.unique(gauss_idx[seen_idx])
            except IndexError:
                # fallback: filter seen_idx to valid range
                seen_idx = seen_idx[(seen_idx >= 0) & (seen_idx < n_gauss)]
                if seen_idx.numel() == 0:
                    return 0.0, 0.0
                visible_v = torch.unique(gauss_idx[seen_idx])

            # filter invalid voxel indices (-1) and clamp to valid voxel range
            num_voxels = int(self.semantic_voxel_map.num_voxels)
            valid_mask = (visible_v >= 0) & (visible_v < num_voxels)
            visible_v = visible_v[valid_mask]
            if visible_v.numel() == 0:
                return 0.0, 0.0

            cam_pos = c2w[:3, 3].to(self.semantic_voxel_map.bbox_min.device)
            device = self.semantic_voxel_map.bbox_min.device

            # Compute visibility weight w_v = mean opacity of seen gaussians per voxel
            try:
                opacities = torch.sigmoid(self.params['logit_opacities'].detach())[:, 0]  # (N_gauss,)
                seen_ops = opacities[seen_idx.to(opacities.device)]                        # (N_seen,)
                seen_vox = gauss_idx[seen_idx].to(opacities.device)                        # (N_seen,) voxel ids
                # scatter-mean: for each voxel sum and count seen-gaussian opacities
                op_sum = torch.zeros(num_voxels, device=opacities.device, dtype=torch.float32)
                op_cnt = torch.zeros(num_voxels, device=opacities.device, dtype=torch.float32)
                seen_vox_clamped = seen_vox.clamp(0, num_voxels - 1)
                op_sum.index_add_(0, seen_vox_clamped, seen_ops.float())
                op_cnt.index_add_(0, seen_vox_clamped, torch.ones_like(seen_ops))
                w_v_map = (op_sum / op_cnt.clamp(min=1.0)).to(device)  # (num_voxels,) mean opacity per voxel
            except Exception:
                w_v_map = None  # fallback: uniform weight = 1.0

            gi_num = 0.0
            gd_num = 0.0
            w_sum = 0.0
            log_c = float(
                torch.log(
                    torch.tensor(
                        max(2, int(self.semantic_voxel_map.n_classes)),
                        device=device,
                        dtype=torch.float32,
                    )
                ).item()
            )

            for v in visible_v.tolist():
                vox_center = self.semantic_voxel_map.get_voxel_center(int(v)).to(device)
                d = cam_pos.to(device) - vox_center
                denom = (d.norm() + 1e-8)
                d_norm = d / denom
                p = int(dir_to_bin(d_norm, nside=self.semantic_voxel_map.nside))

                w_v = float(w_v_map[int(v)].item()) if w_v_map is not None else 1.0
                w_sum += w_v

                try:
                    gi_num += w_v * float(self.semantic_voxel_map.query_heat(int(v), p))
                except Exception:
                    gi_num += 0.0
                try:
                    # pre: 
                    # ent = float(self.semantic_voxel_map.query_entropy(int(v)))
                    # gd_num += w_v * (ent / (log_c + 1e-8))
                    # Use per-voxel directional entropy mean (normalized in [0,1])
                    # instead of full-class voxel entropy. This aggregates the
                    # entropy across active direction bins for voxel v.
                    dir_ent = float(self.semantic_voxel_map._compute_dir_entropy_norm(int(v)))
                    gd_num += w_v * float(dir_ent)
                except Exception:
                    gd_num += 0.0

            if w_sum <= 0.0:
                return 0.0, 0.0

            GI = gi_num / (w_sum + 1e-8)
            GD = gd_num / (w_sum + 1e-8)
            return (GI, GD)

    def load_eval_dataset_with_semantic(self):
        dataset_config = self.config["data"]

        ### gradslam data_fg ###
        if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
        else:
            gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

        if "ignore_bad" not in dataset_config:
            dataset_config["ignore_bad"] = False

        if "use_train_split" not in dataset_config:
            dataset_config["use_train_split"] = True

        if "densification_image_height" not in dataset_config:
            dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
            dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
            self.seperate_densification_res = False
        else:
            if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
                    dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_densification_res = True
            else:
                self.seperate_densification_res = False

        if "tracking_image_height" not in dataset_config:
            dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
            dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
            self.seperate_tracking_res = False
        else:
            if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
                    dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
                self.seperate_tracking_res = True
            else:
                self.seperate_tracking_res = False

        self.dataset_eval = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=self.slam_cfg.dataset_eval_basedir,
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["desired_image_height"],
            desired_width=dataset_config["desired_image_width"],
            device=self.device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
            load_semantics = dataset_config['load_semantics'],
        )
    def init_replica_config_cls2label(self,info_semantic_file, save_json_file):
        with open(info_semantic_file, 'r') as file:
            info_semantic = json.load(file)
        self.num_cls = len(info_semantic["classes"])+1  # for replica, 0 is unknown
        id2label = {}
        for cls in info_semantic["classes"]:
            id2label[str(cls['id'])] =  {"isthing": 1, "name": cls["name"]}

        os.makedirs(os.path.dirname(save_json_file), exist_ok=True)
        with open(save_json_file, "w") as json_file:
            json.dump(id2label, json_file, indent=4)  # indent=4 makes it more readable
        print(f"Semantic Info Dictionary saved to {save_json_file}")

    @torch.no_grad()
    def semantic_annotation(self, input_img: torch.Tensor):
        input_img = (input_img * 255).byte()
        input_img_numpy = input_img.cpu().numpy()
        input_img = Image.fromarray(input_img_numpy)
        class_ids_from_oneformer, class_logits_from_oneformer = oneformer_segmentation(input_img,
                                                                                       self.oneformer_processor,
                                                                                       self.oneformer_model,
                                                                                       self.semantic_device,
                                                                                       num_classes=self.n_cls)
        return class_ids_from_oneformer[0], class_logits_from_oneformer[0]  # (H,W), (H,W,150)

    def init_exploration_map(self, sim2slam: torch.tensor):
        """ initialize exploration map (grid)
    
        Args:
            sim2slam: transformation that transform points from simulation coordinate system to SplaTAM system
    
        Attributes:
            explr_map (ExplorationMap)
            
        """
        self.explr_map = ExplorationMap(
            self.slam_cfg.bbox_bound, 
            self.slam_cfg.bbox_voxel_size, 
            self.device, 
            sim2slam,
            use_xyz_filter=True, 
            xy_sampling_step=self.main_cfg.planner.xy_sampling_step[0], 
            gs_z_levels=self.main_cfg.planner.gs_z_levels[0]
            )
        # Align semantic voxel map to the exploration-map volume after applying sim2slam.
        semantic_bbox = self._transform_bbox_sim_to_slam(
            self.slam_cfg.bbox_bound, sim2slam.to(self.device)
        )
        self.semantic_voxel_map = self._build_semantic_voxel_map(semantic_bbox)
        self.gaussian_voxel_idx = None
        self.voxel_gaussian_lists = {}


    def init_camera_parameters(self):
        """
    
        Attributes:
            params: Splatam parameters
            variables: Splatam variables
            intrinsics: camera intrinsics
            first_frame_w2c: first world to camera pose
            cam
            densify_intrinsics
            densify_cam
            tracking_cam
            
        """
        color, depth, intrinsics, pose = self.dataset_sample[0]
        _, seman = self.semantic_annotation(color)
        seman = seman.to(self.device)

        if self.seperate_densification_res:
            # Initialize Parameters, Canonical & Densification Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam, \
                densify_intrinsics, densify_cam = initialize_first_timestep(self.dataset_sample, seman, self.num_frames,
                                                                                 self.config['scene_radius_depth_ratio'],
                                                                                 self.config['mean_sq_dist_method'],
                                                                                 densify_dataset=self.densify_dataset_sample,
                                                                                 gaussian_distribution=self.config['gaussian_distribution'],
                                                                                 TOPK=self.topk, num_classes=self.n_cls)
            # return params, variables, intrinsics, first_frame_w2c, cam, \
            #     densify_intrinsics, densify_cam
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        else:
            # Initialize Parameters & Canoncial Camera parameters
            params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(self.dataset_sample, seman, self.num_frames,
                                                                                            self.config['scene_radius_depth_ratio'],
                                                                                            self.config['mean_sq_dist_method'],
                                                                                            gaussian_distribution=self.config['gaussian_distribution'],
                                                                                            TOPK=self.topk, num_classes=self.n_cls)
            # return params, variables, intrinsics, first_frame_w2c, cam
            self.densify_intrinsics = intrinsics
            self.densify_cam = cam
        
        if self.seperate_tracking_res:
            self.tracking_cam = setup_camera(self.tracking_color.shape[2], self.tracking_color.shape[1], 
                                        self.tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy(),
                                             num_channels = self.n_cls)

        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam
    
    @torch.no_grad()
    def render(self, c2w: torch.Tensor, return_silhouette: bool = False) -> Tuple[torch.Tensor, ...]:
        ''' render rgb, mask, and depth based on a given pose
        Args:
            c2w: [4,4]. camera-to-world pose, in SplaTAM system
        
        Returns:
            im: (C,H,W) # render image
            depth: (1,H,W) # render depth
            mask: (1,H,W) valid rendering mask
        '''
        cam = self.cam
        first_frame_w2c = self.first_frame_w2c
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)
        sil_thres = self.config['mapping']['sil_thres']
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        rendervar = transformed_params2rendervar(params, transformed_gaussians)
        depth_sil_rendervar = transformed_params2depthplussilhouette(params, first_frame_w2c, 
                                                                     transformed_gaussians)
    
        im, radii, _, = Renderer(raster_settings=cam)(**rendervar)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**depth_sil_rendervar)
        rastered_depth = depth_sil[0, :, :].unsqueeze(0)
        valid_depth_mask = (depth_sil[0:1] > 0)
        silhouette = depth_sil[1, :, :]
        presence_sil_mask = (silhouette > sil_thres)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        seen = (radii > 0)
        if return_silhouette:
            return im, rastered_depth, valid_depth_mask, seen, silhouette
        return im, rastered_depth, valid_depth_mask, seen

    @torch.no_grad()
    # TODO: MERGE TO RGB render
    def render_semantic(self, c2w: torch.Tensor, seen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' render semantic class and prob based on a given pose
        Args:
            c2w: [4,4]. camera-to-world pose, in SplaTAM system

        Returns:
            class_ids: (H,W) # semantic class ids
            logits: (C,H,W) # semantic probability logits
        '''
        cam = self.cam
        first_frame_w2c = self.first_frame_w2c
        gt_w2c = torch.linalg.inv(c2w)
        cam_params = self.initialize_cam_params(1)
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
            cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        variables = self.variables
        cam_trans_og = self.params['cam_trans']
        cam_rot_og = self.params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']
        transformed_gaussians = transform_to_frame(params, 0,
                                                   gaussians_grad=False,
                                                   camera_grad=False)

        # Initialize Render Variables
        seman_rendervar = transformed_params2semrendervar_sparse(params, transformed_gaussians, seen)
        sparse_cam = set_camera_sparse(cam=cam, cls_ids=variables['seman_cls_ids'])
        logits, _, = SEMRenderer_sparse(raster_settings=sparse_cam)(**seman_rendervar)

        self.params['cam_trans'] = cam_trans_og
        self.params['cam_unnorm_rots'] = cam_rot_og
        class_id = logits.argmax(dim=0)
        return class_id, logits

    @torch.no_grad()
    def render_semantic_visualization(self, c2w: torch.Tensor, seen: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Render semantic class/prob using the same dense path as eval visualization.'''
        cam = self.cam
        gt_w2c = torch.linalg.inv(c2w)

        cam_params = self.initialize_cam_params(1)
        rel_w2c_rot = gt_w2c[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = gt_w2c[:3, 3].detach()
        cam_params['cam_unnorm_rots'][..., 0] = rel_w2c_rot_quat
        cam_params['cam_trans'][..., 0] = rel_w2c_tran

        params = self.params
        variables = self.variables
        cam_trans_og = params['cam_trans']
        cam_rot_og = params['cam_unnorm_rots']
        params['cam_trans'] = cam_params['cam_trans']
        params['cam_unnorm_rots'] = cam_params['cam_unnorm_rots']

        transformed_gaussians = transform_to_frame(
            params,
            0,
            gaussians_grad=False,
            camera_grad=False,
        )

        seman_rendervar = transformed_params2semrendervar(params, variables, transformed_gaussians, seen)
        logits, _, = SEMRenderer(raster_settings=cam)(**seman_rendervar)
        logits = torch.nan_to_num(logits, nan=0.0)
        logits[logits < 0] = 0.0

        params['cam_trans'] = cam_trans_og
        params['cam_unnorm_rots'] = cam_rot_og

        class_id = logits.argmax(dim=0)
        return class_id, logits

    def plot_render_depth(self, c2w: torch.Tensor):
        """ plot rendered depth at the given pose
    
        Args:
            c2w: [4,4], camera-to-world, RDF
        """
        _, depth, _, _ = self.render(c2w)
        depth = depth[0].detach().cpu().numpy()
        plt.imshow(depth)
        plt.show()
    
    def plot_render_rgb(self, c2w: torch.Tensor):
        """ plot rendered RGB at the given pose

        Args:
            c2w: [4,4], camera-to-world, RDF
        """
        im, _, _, _ = self.render(c2w)
        im = im.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(im)
        plt.show()

    def online_recon_step(self, 
                          time_idx        : int,
                          color           : torch.Tensor,
                          depth           : torch.Tensor,
                          c2w             : torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          ) -> List:
        ''' Run one step of the co-slam process.

        Args:
            time_idx        : Current frame step
            color           : color,        [H,W,3]
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
            only_use_global_keyframe: post-refinement stage
        
        Returns:
        '''
        if time_idx == 0:
            self.init_camera_parameters()

        seg_img = color.clone().to(self.semantic_device, non_blocking=True)
        _, seman = self.semantic_annotation(seg_img)
        self.update_gs_map(time_idx, color, depth, seman, c2w, force_map_update, dont_add_kf, only_use_global_keyframe)
        if self.slam_cfg.enable_active_planning:
            self.update_explr_map(time_idx, depth, c2w, force_map_update)

    def update_global_keyframe_set_completeness(self, depth, c2w, thre, 
                                   time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    ### get render depth and new region mask (valid number) ###
                    ### get new region mask ###
                    new_pixel_num = ((depth > 0)*(~self.render(c2w)[2])).sum()

                    ### determine if is global keyframe ###
                    _, h, w = depth.shape
                    new_pixel_ratio = new_pixel_num / (h * w)
                    is_global_kf = new_pixel_ratio  > thre

                    ### add the index to global keyframe index ###
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality(self, color, depth, c2w, color_thre, depth_thre,
                                   time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        if time_idx in self.global_keyframe_time_indices:
            return 
        
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    render_color, render_depth, _, _ = self.render(c2w)
                    valid_depth_mask = depth > 0
                    color_ig = calc_psnr(render_color*valid_depth_mask, color*valid_depth_mask).mean()
                    depth_ig = (torch.abs(render_depth*valid_depth_mask - depth*valid_depth_mask)/(depth+1e-8)).sum() / valid_depth_mask.sum() 

                    ### FIXME: update is_global_kf condition ###
                    is_global_kf = color_ig < color_thre


                    ### add the index to global keyframe index ###
                    if is_global_kf or time_idx == 0:
                        self.global_keyframe_indices.append(len(self.keyframe_list))
                        self.global_keyframe_time_indices.append(time_idx)

    def update_global_keyframe_set_quality_rel(self, 
                                            #   color, depth, c2w, color_thre, depth_thre,
                                            #     time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
                                                ):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        ### Render All KFs, excluding last few (5) keyframes (due to overfitting) ###
        color_igs = []
        seman_igs = []
        for kf in self.keyframe_list[:-5]:
            c2w = torch.inverse(kf['est_w2c']) # c2w
            color, _, valid_mask, seen = self.render(c2w)
            # _, pred_logits = self.render_semantic(c2w,seen)

            ### compute REFINE I.G. ###
            valid_depth_mask = kf['depth'] > 0.2 # FIXME: 0.2 is near culling range
            color_ig = calc_psnr(color*valid_depth_mask, kf['color']*valid_depth_mask).mean()
            color_igs.append(color_ig)

            seg_img = kf['color'].clone().permute(1, 2, 0).to(self.semantic_device)
            pseudo_cls, pseudo_logits = self.semantic_annotation(seg_img)
            pseudo_logits = pseudo_logits.to(self.device)
            entropy = calc_shannon_entropy(pseudo_logits, dim=-1)
            pseudo_cls = pseudo_cls.to(self.device)
            unidentify = ((pseudo_cls == 0) | (entropy > 3.0)) & valid_depth_mask
            seman_ig = (~unidentify).sum()
            seman_igs.append(seman_ig)

        selected_kf_idxs = []
        if len(color_igs) > 0:
            ### compute weighted Refinement I.G., weighted by distance ###
            color_igs = torch.stack(color_igs).float()

            ### Rank based on PSNR ###
            color_thre = torch.quantile(color_igs, self.slam_cfg.global_keyframe.quality_perc_thre/100.)
            color_kf_idxs = torch.where(color_igs <= color_thre)[0]
            selected_kf_idxs = color_kf_idxs

        # TODO: add semantic ig
        if len(seman_igs) > 0:
            seman_igs = torch.stack(seman_igs).float()

            ### Rank based on Entropy  ###
            seman_thre = torch.quantile(seman_igs, self.slam_cfg.global_keyframe.quality_perc_thre / 100.)
            seman_kf_idxs = torch.where(seman_igs <= seman_thre)[0]
            selected_kf_idxs = torch.unique(torch.cat((selected_kf_idxs, seman_kf_idxs)))

        if len(selected_kf_idxs) > 0:
            ### add low quality frames to Global KF (if they are not there yet) ###
            new_kf = [elem.item() for elem in selected_kf_idxs if elem not in self.global_keyframe_indices]
            self.global_keyframe_indices.extend(new_kf)
            new_kf_time_indices = [self.keyframe_time_indices[i] for i in new_kf]
            self.global_keyframe_time_indices.extend(new_kf_time_indices)

    @torch.no_grad()
    def update_explr_map(self,
                           time_idx        : int,
                           depth           : torch.Tensor,
                           c2w             : torch.Tensor,
                           force_map_update: bool = False,
                                                 ) -> torch.Tensor : 
        ''' Run one step of the co-slam process.

        Args:
            time_idx        : Current frame step
            depth           : depth map,    [H,W]
            c2w             : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
        
        Attributes:
            explr_map: update exploration map
        '''
        config = self.config
        depth = depth.to(self.device)
        c2w = c2w.to(self.device)
        changed_occ_voxels = torch.empty((0, 3), dtype=torch.long, device=self.device)
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0 or force_map_update:
            changed_occ_voxels = self.explr_map.update_from_depth_map(
                depth, 
                self.intrinsics, 
                torch.inverse(c2w),
                self.slam_cfg.surface_dist_thre,
                self.slam_cfg.get("find_free_indices_bs", 10000)
                )

            if hasattr(self, 'semantic_voxel_map'):
                occ_dirty_radius = int(getattr(self.slam_cfg, 'sem_voxel_occ_dirty_radius', 1))
                self.semantic_voxel_map.mark_occupancy_dirty(
                    occ_voxels=changed_occ_voxels,
                    occ_origin=self.explr_map.origin,
                    occ_voxel_size=self.explr_map.voxel_size,
                    sim2slam=self.explr_map.sim2slam,
                    radius=occ_dirty_radius,
                )

                if bool(getattr(self.slam_cfg, 'sem_voxel_feasible_event_driven', False)):
                    slam2sim = self.explr_map.slam2sim.to(self.device)
                    full_every = int(getattr(self.slam_cfg, 'sem_voxel_feasible_full_refresh_every', 0))
                    full_refresh = bool(full_every > 0 and (time_idx % full_every == 0))
                    self.semantic_voxel_map.update_feasible_mask(
                        occ_grid=self.explr_map.occupancy_grid,
                        occ_origin=self.explr_map.origin,
                        occ_voxel_size=self.explr_map.voxel_size,
                        slam2sim=slam2sim,
                        check_depth=int(getattr(self.slam_cfg, 'sem_voxel_feasible_check_depth', 3)),
                        full_refresh=full_refresh,
                    )

        return changed_occ_voxels


    def update_gs_map(self, 
                          time_idx: int,
                          color   : torch.Tensor,
                          depth   : torch.Tensor,
                          seman   : torch.Tensor,
                          c2w     : torch.Tensor,
                          force_map_update: bool = False,
                          dont_add_kf: bool = False,
                          only_use_global_keyframe: bool = False,
                          ) -> List:
        ''' Run one step of the splatam process. Update GS

        Args:
            time_idx: Current frame step
            color   : color,        [H,W,3]
            depth   : depth map,    [H,W]
            seman   : semantic map,    [H,W,num_classes] coco use 133 classes
            c2w     : pose. Format: RUB camera-to-world, [4,4]
            force_map_update: run map update if true
        '''
        ### get self variables ###

        params = self.params
        variables = self.variables
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        cam = self.cam
        seperate_densification_res = self.seperate_densification_res
        if seperate_densification_res:
            densify_intrinsics = self.densify_intrinsics
            densify_cam = self.densify_cam
        config = self.config
        gt_w2c_all_frames = self.gt_w2c_all_frames
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
            wandb_mapping_step = self.wandb_mapping_step
            wandb_time_step = self.wandb_time_step
        eval_dir = self.eval_dir
        seperate_tracking_res = self.seperate_tracking_res
        if seperate_tracking_res:
            tracking_cam = self.tracking_cam
            tracking_intrinsics = self.tracking_intrinsics
        keyframe_list = self.keyframe_list
        num_frames = self.num_frames
        keyframe_time_indices = self.keyframe_time_indices


        ### Process poses ###
        gt_w2c = torch.linalg.inv(c2w)


        # Process RGB-D Data
        color = color.permute(2, 0, 1)
        color = color.to(self.device)
        depth = depth.unsqueeze(0)
        depth = depth.to(self.device)
        seman = seman.permute(2, 0, 1)  # C,H,W
        seman = seman.to(self.device)

        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx


        # Initialize Mapping Data for selected frame
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'seman': seman, 'id': iter_time_idx, 'intrinsics': intrinsics,
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # # Initialize Data for Tracking
        if seperate_tracking_res:
            ### Load tracking data ###
            tracking_h, tracking_w = self.config['data']["tracking_image_height"], self.config['data']["tracking_image_width"]
            tracking_color = F.interpolate(color.unsqueeze(0), (tracking_h, tracking_w), mode='bilinear')[0]
            tracking_depth = F.interpolate(depth.unsqueeze(0), (tracking_h, tracking_w), mode='nearest')[0]
            
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data

        # Optimization Iterations
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0:
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])

        ##################################################
        ### Tracking
        ##################################################
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']:
            # Reset Optimizer & Learning Rates for tracking
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)
            # Keep Track of Best Candidate Rotation & Translation
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
            current_min_loss = float(1e20)
            # Tracking Optimization
            iter = 0
            do_continue_slam = False
            num_iters_tracking = config['tracking']['num_iters']
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
            while True:
                iter_start_time = time.time()
                # Loss for current frame
                loss, variables, losses = get_loss_with_seman(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                # Backprop
                loss.backward()
                # Optimizer Update
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # Save the best candidate rotation & translation
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params,variables, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params,variables, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.tracking_iter_time_sum += iter_end_time - iter_start_time
                self.tracking_iter_time_count += 1
                # Check if we should stop tracking
                iter += 1
                if iter == num_iters_tracking:
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        # Update the runtime numbers
        tracking_end_time = time.time()
        self.tracking_frame_time_sum += tracking_end_time - tracking_start_time
        self.tracking_frame_time_count += 1

        if (time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0) and not config['tracking']['use_gt_poses']:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, variables,tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, variables, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')

        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            self.update_global_keyframe_set_completeness(
                depth, c2w, 
                self.slam_cfg.global_keyframe.completeness_thre, 
                time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
            )

        ##################################################
        ### Densification & KeyFrame-based Mapping
        ##################################################
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0 or force_map_update:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                # Setup Data for Densification
                if seperate_densification_res:
                    # resize RGBD frames for densification
                    densify_h, densify_w = self.config['data']["densification_image_height"], self.config['data']["densification_image_width"]
                    densify_color = F.interpolate(color.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]
                    densify_depth = F.interpolate(depth.unsqueeze(0), (densify_h, densify_w), mode='nearest')[0]
                    densify_seman = F.interpolate(seman.unsqueeze(0), (densify_h, densify_w), mode='bilinear')[0]
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'seman': densify_seman,'id': time_idx,
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data

                # Add new Gaussians to the scene based on the Silhouette
                params, variables = add_new_gaussians_with_seman(params, variables, densify_curr_data,
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'], config['gaussian_distribution'],self.topk)
                post_num_pts = params['means3D'].shape[0]
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                ##################################################
                ### Select Keyframes for Mapping
                ##################################################
                ### overlap keyframes ###
                num_keyframes = config['mapping_window_size']-2
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                if PRINT_INFO:
                    print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")
                    if self.slam_cfg.use_global_keyframe:
                        global_keyframe_time_indices = [frame_idx for frame_idx in self.global_keyframe_time_indices if frame_idx != time_idx]
                        print(f"\nGlobal Keyframes at Frame {time_idx}: {global_keyframe_time_indices}")
                
            # Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)


            # Mapping
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            sem_voxel_updated_this_frame = False
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                ##################################################
                ### frame selection for map update
                ##################################################
                ### Overlap Keyframe ###

                if only_use_global_keyframe or (self.slam_cfg.use_global_keyframe and iter > num_iters_mapping // 2):
                    ### Global Keyframe ###
                    if len(self.global_keyframe_indices) == 1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_seman = seman
                        iter_crop_mask = None
                    else:
                        selected_rand_keyframe_idx = np.random.choice(self.global_keyframe_indices[:-1])
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color'] # C,H,W
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_seg_img = iter_color.permute(1,2,0).clone().to(self.semantic_device, non_blocking=True)
                        _, iter_seman = self.semantic_annotation(iter_seg_img)
                        iter_seman = iter_seman.permute(2,0,1).to(self.device, non_blocking=True)
                        if 'crop_mask' in keyframe_list[selected_rand_keyframe_idx].keys():
                            iter_crop_mask = keyframe_list[selected_rand_keyframe_idx]['crop_mask']
                        else:
                            iter_crop_mask = get_uncert_mask(iter_seman,filter_pct=0.1,thres=self.slam_cfg['uncert_mask_thres'])
                            keyframe_list[selected_rand_keyframe_idx]['crop_mask'] = iter_crop_mask
                else:
                    rand_idx = np.random.randint(0, len(selected_keyframes))
                    selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                    if selected_rand_keyframe_idx == -1:
                        # Use Current Frame Data
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                        iter_seman = seman
                        iter_crop_mask = None
                    else:
                        # Use Keyframe Data
                        iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                        iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                        iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                        iter_seg_img = iter_color.permute(1, 2, 0).clone().to(self.semantic_device, non_blocking=True)
                        _, iter_seman = self.semantic_annotation(iter_seg_img)
                        iter_seman = iter_seman.permute(2,0,1).to(self.device, non_blocking=True)
                        if 'crop_mask' in keyframe_list[selected_rand_keyframe_idx].keys():
                            iter_crop_mask = keyframe_list[selected_rand_keyframe_idx]['crop_mask']
                        else:
                            iter_crop_mask = get_uncert_mask(iter_seman, filter_pct=0.1,thres= self.slam_cfg['uncert_mask_thres'])
                            keyframe_list[selected_rand_keyframe_idx]['crop_mask'] = iter_crop_mask

                
                iter_gt_w2c = self.gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'seman': iter_seman, 'id': iter_time_idx,
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                if iter_crop_mask is not None:
                    iter_data['crop_mask'] = iter_crop_mask

                # Optional: enable only for OOM debugging; frequent empty_cache can reduce throughput.
                if bool(getattr(self.slam_cfg, 'empty_cache_during_mapping_iter', False)):
                    torch.cuda.empty_cache()
                # Loss for current frame
                need_semantic_artifacts = bool(
                    self.slam_cfg.enable_active_planning and (iter_time_idx == time_idx) and (not sem_voxel_updated_this_frame)
                )
                if need_semantic_artifacts:
                    loss, variables, losses, render_artifacts = get_loss_with_seman(
                        params,
                        iter_data,
                        variables,
                        iter_time_idx,
                        config['mapping']['loss_weights'],
                        config['mapping']['use_sil_for_loss'],
                        config['mapping']['sil_thres'],
                        config['mapping']['use_l1'],
                        config['mapping']['ignore_outlier_depth_loss'],
                        mapping=True,
                        lambda_hel=self.slam_cfg['lambda_hel'],
                        lambda_cosine=self.slam_cfg['lambda_cosine'],
                        return_render_artifacts=True,
                    )
                else:
                    loss, variables, losses = get_loss_with_seman(
                        params,
                        iter_data,
                        variables,
                        iter_time_idx,
                        config['mapping']['loss_weights'],
                        config['mapping']['use_sil_for_loss'],
                        config['mapping']['sil_thres'],
                        config['mapping']['use_l1'],
                        config['mapping']['ignore_outlier_depth_loss'],
                        mapping=True,
                        lambda_hel=self.slam_cfg['lambda_hel'],
                        lambda_cosine=self.slam_cfg['lambda_cosine'],
                    )
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                # Backprop
                torch.nn.utils.clip_grad_norm_([v for v in self.params.values() if isinstance(v, torch.nn.Parameter)], max_norm=100.0)
                loss.backward()

                if need_semantic_artifacts:
                    with torch.no_grad():
                        H_i, W_i = iter_depth.shape[-2], iter_depth.shape[-1]

                        intr_i = intrinsics.to(device=self.device, dtype=torch.float32) if torch.is_tensor(intrinsics) else torch.as_tensor(intrinsics, device=self.device, dtype=torch.float32)
                        fx_i, fy_i = intr_i[0, 0], intr_i[1, 1]
                        cx_i, cy_i = intr_i[0, 2], intr_i[1, 2]

                        # Cache pixel grids by resolution/device to avoid rebuilding every frame.
                        grid_cache = getattr(self, '_sem_backproj_grid_cache', None)
                        if grid_cache is None:
                            grid_cache = {}
                            self._sem_backproj_grid_cache = grid_cache
                        dev_idx = self.device.index if self.device.index is not None else -1
                        grid_key = (int(H_i), int(W_i), self.device.type, int(dev_idx))
                        if grid_key in grid_cache:
                            uu, vv = grid_cache[grid_key]
                        else:
                            u_axis = torch.arange(W_i, device=self.device, dtype=torch.float32)
                            v_axis = torch.arange(H_i, device=self.device, dtype=torch.float32)
                            uu, vv = torch.meshgrid(u_axis, v_axis, indexing='xy')
                            grid_cache[grid_key] = (uu, vv)

                        z_i = iter_depth.squeeze(0).float().to(self.device)
                        x_cam_i = (uu - cx_i) * z_i / fx_i
                        y_cam_i = (vv - cy_i) * z_i / fy_i
                        pts_cam_i = torch.stack([x_cam_i.reshape(-1), y_cam_i.reshape(-1), z_i.reshape(-1)], dim=-1)
                        c2w_i = c2w.to(device=self.device, dtype=torch.float32) if torch.is_tensor(c2w) else torch.as_tensor(c2w, device=self.device, dtype=torch.float32)
                        pts_w_i = pts_cam_i @ c2w_i[:3, :3].T + c2w_i[:3, 3]

                        probs_obs_i = iter_seman.permute(1, 2, 0).reshape(-1, self.n_cls).float().to(self.device)
                        probs_rend_i = render_artifacts['seman_im'].permute(1, 2, 0).reshape(-1, self.n_cls).float().to(self.device)
                        sil_i = render_artifacts['silhouette'].reshape(-1).float().to(self.device)
                        valid_i = z_i.reshape(-1) > 0

                        # Optional speed knobs for semantic-voxel update.
                        stride_i = int(getattr(self.slam_cfg, 'sem_voxel_update_stride', 1))
                        if stride_i > 1:
                            sample_mask_i = torch.zeros_like(valid_i)
                            sample_mask_i[::stride_i] = True
                            valid_i = valid_i & sample_mask_i

                        max_pix_i = int(getattr(self.slam_cfg, 'sem_voxel_update_max_pixels', 0))
                        if max_pix_i > 0:
                            idx_i = torch.where(valid_i)[0]
                            if idx_i.numel() > max_pix_i:
                                step_i = max(1, int(idx_i.numel() // max_pix_i))
                                keep_i = idx_i[::step_i][:max_pix_i]
                                cap_mask_i = torch.zeros_like(valid_i)
                                cap_mask_i[keep_i] = True
                                valid_i = valid_i & cap_mask_i

                        self.semantic_voxel_map.update_from_frame(
                            pts_w_i[valid_i],
                            probs_obs_i[valid_i],
                            c2w_i[:3, 3],
                            probs_render=probs_rend_i[valid_i],
                            silhouette=sil_i[valid_i],
                        )
                        sem_voxel_updated_this_frame = True

                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']:
                        params, variables = prune_gaussians_w_semantic(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']:
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, variables, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, variables, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'],
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                self.mapping_iter_time_sum += iter_end_time - iter_start_time
                self.mapping_iter_time_count += 1

            dense_shape = (params['log_scales'].shape[0],133)
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            self.mapping_frame_time_sum += mapping_end_time - mapping_start_time
            self.mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, variables,curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'],
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True,eval_dir=self.eval_dir)
                        else:
                            report_progress(params, variables, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'],
                                            eval_dir=self.eval_dir,
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, variables, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')

        ##################################################
        ### update global keyframe
        ##################################################
        if self.slam_cfg.use_global_keyframe and not(only_use_global_keyframe):
            quality_method = self.slam_cfg.global_keyframe.get("quality_method", "absolute")
            if quality_method == "absolute":
                self.update_global_keyframe_set_quality(
                    color, depth, c2w, 
                    self.slam_cfg.global_keyframe.color_thre, 
                    self.slam_cfg.global_keyframe.depth_thre, 
                    time_idx, curr_gt_w2c, dont_add_kf, num_frames, force_map_update, config
                )
            elif quality_method == "relative":
                if time_idx > 0 and time_idx % self.slam_cfg.global_keyframe.quality_freq == 0:
                    self.update_global_keyframe_set_quality_rel()
            else:
                raise NotImplementedError
        
        # Add frame to keyframe list
        if not(dont_add_kf):
            if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                        (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()) or force_map_update:
                with torch.no_grad():
                    # Get the current estimated rotation & translation
                    curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                    curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                    curr_w2c = torch.eye(4).cuda().float()
                    curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                    curr_w2c[:3, 3] = curr_cam_tran
                    # Initialize Keyframe Info
                    curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                    # Add to keyframe list
                    keyframe_list.append(curr_keyframe)
                    keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, variables, ckpt_output_dir, time_idx)
            save_semantic_ply(params,variables,ckpt_output_dir, time_idx, self.n_cls)
            save_rgb_ply(params,ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))

        if time_idx % 100 == 0 and hasattr(self, 'semantic_voxel_map'):
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_voxel_heat_ply(self.semantic_voxel_map, ckpt_output_dir, ply_name=f"voxel_heat_{time_idx:04}.ply")
            save_voxel_heat_ply(
                self.semantic_voxel_map,
                ckpt_output_dir,
                ply_name=f"directional_hbar_{time_idx:04}.ply",
                scalar_mode="h_bar",
            )
        
        # Increment WandB Time Step
        if config['use_wandb']:
            self.wandb_time_step += 1

        if bool(getattr(self.slam_cfg, 'empty_cache_after_frame', False)):
            torch.cuda.empty_cache()
        
        ##################################################
        ### update self variables
        ##################################################
        self.params = params
        self.variables = variables
        self.intrinsics = intrinsics
        self.first_frame_w2c = first_frame_w2c
        self.cam = cam
        self.seperate_densification_res = seperate_densification_res
        if self.seperate_densification_res:
            self.densify_intrinsics = densify_intrinsics
            self.densify_cam = densify_cam
        self.config = config
        self.gt_w2c_all_frames = gt_w2c_all_frames
        if self.config['use_wandb']:
            self.wandb_run = wandb_run
            self.wandb_mapping_step = wandb_mapping_step
            self.wandb_time_step = wandb_time_step
        self.seperate_tracking_res = seperate_tracking_res
        if self.seperate_tracking_res:
            self.tracking_cam = tracking_cam
            self.tracking_intrinsics = tracking_intrinsics
        
        self.keyframe_list = keyframe_list
        self.num_frames = num_frames
        self.keyframe_time_indices = keyframe_time_indices
        # 更新gaussian 和 voxel的索引
        if time_idx % self.slam_cfg.sem_voxel_rebuild_freq == 0: 
            self.rebuild_gaussian_voxel_index()
            # 同步更新可行性遮罩（只對 surface/boundary voxel，與 rebuild 同頻）
            if hasattr(self, 'explr_map') and hasattr(self, 'semantic_voxel_map'):
                slam2sim = self.explr_map.slam2sim.to(self.device)
                event_driven = bool(getattr(self.slam_cfg, 'sem_voxel_feasible_event_driven', False))
                self.semantic_voxel_map.update_feasible_mask(
                    occ_grid=self.explr_map.occupancy_grid,
                    occ_origin=self.explr_map.origin,
                    occ_voxel_size=self.explr_map.voxel_size,
                    slam2sim=slam2sim,
                    check_depth=int(getattr(self.slam_cfg, 'sem_voxel_feasible_check_depth', 3)),
                    full_refresh=not event_driven,
                )

    def print_and_save_result(self, eval_dir_suffix="", is_prune_gaussians=False, ignore_first_frame=False):
        """ evaluate rendering results and save result
        """
        ### get self variables ###
        params = self.params.copy()
        variables = self.variables.copy()
        intrinsics = self.intrinsics
        first_frame_w2c = self.first_frame_w2c
        tracking_iter_time_sum = self.tracking_iter_time_sum
        tracking_frame_time_sum = self.tracking_frame_time_sum
        tracking_iter_time_count = self.tracking_iter_time_count
        tracking_frame_time_count = self.tracking_frame_time_count
        mapping_iter_time_sum = self.mapping_iter_time_sum
        mapping_frame_time_sum = self.mapping_frame_time_sum
        mapping_iter_time_count = self.mapping_iter_time_count
        mapping_frame_time_count = self.mapping_frame_time_count
        config = self.config
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        dataset_config = self.config['data']
        gt_w2c_all_frames = self.gt_w2c_all_frames
        keyframe_time_indices = self.keyframe_time_indices
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        ### prune gaussians ###
        if is_prune_gaussians:
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)
            params, variables = prune_gaussians_w_semantic(params, variables, optimizer, 0, config['mapping']['pruning_dict'])


        # Compute Average Runtimes
        if tracking_iter_time_count == 0:
            tracking_iter_time_count = 1
            tracking_frame_time_count = 1
        if mapping_iter_time_count == 0:
            mapping_iter_time_count = 1
            mapping_frame_time_count = 1
        tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
        tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
        mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
        mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
        print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
        print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
        print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
        print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
        if config['use_wandb']:
            wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                        "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                        "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                        "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                        "Final Stats/step": 1})
        
        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(self,dataset, params,variables, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)
                eval_semantic(self,dataset, params,variables, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)
            else:
                eval(self,dataset, params,variables, len(dataset), eval_dir,sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'],
                    ignore_first_frame=ignore_first_frame)
                eval_semantic(self, dataset, params, variables, len(dataset), eval_dir, sil_thres=config['mapping']['sil_thres'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'],
                     ignore_first_frame=ignore_first_frame)

        # Add Camera Parameters to Save them
        params['timestep'] = variables['timestep']
        params['intrinsics'] = intrinsics.detach().cpu().numpy()
        params['w2c'] = first_frame_w2c.detach().cpu().numpy()
        params['org_width'] = dataset_config["desired_image_width"]
        params['org_height'] = dataset_config["desired_image_height"]
        params['gt_w2c_all_frames'] = []
        for gt_w2c_tensor in gt_w2c_all_frames:
            params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
        params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
        params['keyframe_time_indices'] = np.array(keyframe_time_indices)
        params['seman_cls_ids'] = variables['seman_cls_ids'].detach().cpu().contiguous().numpy()

        # Save Parameters
        results_dir = os.path.join(self.results_dir, eval_dir_suffix) if eval_dir_suffix else self.results_dir
        os.makedirs(results_dir, exist_ok=True)
        save_params(params, results_dir)
        if hasattr(self, 'semantic_voxel_map'):
            save_voxel_heat_ply(self.semantic_voxel_map, results_dir)
            save_voxel_heat_ply(
                self.semantic_voxel_map,
                results_dir,
                ply_name="directional_hbar_final.ply",
                scalar_mode="h_bar",
            )

    def load_params(self, stage="final"):
        """ load checkpoint parameters, overrides base class to also set self.variables
        Attributes:
            params: SplaTAM parameters
            variables: SemSplatam variables (seman_cls_ids, n_cls)
        """
        config = self.config
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"{stage}/params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        if 'seman_cls_ids' in params:
            seman_cls_ids = params.pop('seman_cls_ids').to(torch.long)
        else:
            # Derive from semantic_logits if not saved (argmax over class dimension)
            seman_cls_ids = torch.argmax(params['semantic_logits'], dim=1).to(torch.long)
        self.variables = {'seman_cls_ids': seman_cls_ids,
                          'n_cls': self.n_cls}
        self.params = params

    def load_params_by_step(self, step=1100,stage='final'):
        """ load checkpoint parameters
        Attributes:
            params: SplaTAM parameters
        """
        ### load self variables ###
        config = self.config
        checkpoint_time_idx = step
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        if checkpoint_time_idx == 0:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"{stage}/params.npz")
        else:
            ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        self.variables = {'seman_cls_ids': params.pop('seman_cls_ids').to(torch.long),
                          'n_cls': self.n_cls}
        self.params = params


    def eval_result(self, eval_dir_suffix="", ignore_first_frame = False, save_frames=False):
        """ evaluate rendering results
            
        """
        ### get self variables ###
        params = self.params
        variables = self.variables
        config = self.config
        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        # eval_dir = self.eval_dir
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval(self,dataset, params, variables, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame)
            else:
                eval(self,dataset, params, variables, num_frames, eval_dir,sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame, save_frames=save_frames)
        return

    def eval_semantic_result(self, eval_dir_suffix="", ignore_first_frame=False, save_frames=False):
        """ evaluate rendering results

        """
        ### get self variables ###
        params = self.params
        variables = self.variables
        config = self.config

        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        save_semantic_ply(params, variables, eval_dir, 2000, self.n_cls)

        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval_semantic(self, dataset, params, variables, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame)
            else:
                eval_semantic(self, dataset, params, variables, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=config['eval_every'], ignore_first_frame=ignore_first_frame, save_frames=save_frames)
        return

    def eval_semantic_result_mp3d(self, eval_dir_suffix="", ignore_first_frame=False, save_frames=False):
        """ evaluate rendering results

        """
        ### get self variables ###
        params = self.params
        variables = self.variables
        config = self.config

        if self.config['use_wandb']:
            wandb_run = self.wandb_run
        # dataset = self.dataset_sample
        dataset = self.dataset_eval
        num_frames = self.num_frames
        eval_dir = self.eval_dir + "_" + eval_dir_suffix if eval_dir_suffix else self.eval_dir

        save_semantic_ply(params, variables, eval_dir, 2000, self.n_cls)

        # Evaluate Final Parameters
        with torch.no_grad():
            if config['use_wandb']:
                eval_semantic_mp3d(self, dataset, params, variables, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=1, ignore_first_frame=ignore_first_frame)
            else:
                eval_semantic_mp3d(self, dataset, params, variables, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                     mapping_iters=config['mapping']['num_iters'],
                     add_new_gaussians=config['mapping']['add_new_gaussians'],
                     eval_every=1, ignore_first_frame=ignore_first_frame, save_frames=save_frames)
        return
    
    def update_dict_recursive(self, dict1, dict2):
        """
        Recursively updates the values of dict1 with the values from dict2.
        If a key in dict2 corresponds to a nested dictionary, it recursively updates dict1.

        Args:
            dict1 (Dict[Any, Any]): The dictionary to be updated.
            dict2 (Dict[Any, Any]): The dictionary with the new values.

        Returns:
            Dict[Any, Any]: The updated dict1 with values from dict2.
        """
        for key, value in dict2.items():
            if key not in dict1:
                ### Raise an error if key in dict2 is not found in dict1
                raise KeyError(f"Key '{key}' not found in dict1.")
    
            if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
                ### If both dict1[key] and dict2[key] are dictionaries, recursively update them
                dict1[key] = self.update_dict_recursive(dict1[key], value)
            else:
                ### Otherwise, update the value in dict1 with the value from dict2
                dict1[key] = value
        return dict1

    def override_config(self, config: Dict) -> Dict:
        """ override configs 
        """
        ### override from main_cfg ###
        config["data"]["sequence"] = self.main_cfg.general.scene
        config["workdir"] = os.path.join(self.main_cfg.dirs.result_dir, "splatam")
        config["run_name"] = ""
        if self.main_cfg.general.dataset == 'Replica':
            config["data"]['gradslam_data_cfg'] = os.path.join("third_parties/splatam", config["data"]['gradslam_data_cfg'])

        ## specific override ##
        config = self.update_dict_recursive(config, self.main_cfg.slam.override)

        ### Spltam config update ###
        # Print Config
        print("Loaded Config:")
        if "use_depth_loss_thres" not in config['tracking']:
            config['tracking']['use_depth_loss_thres'] = False
            config['tracking']['depth_loss_thres'] = 100000
        if "visualize_tracking_loss" not in config['tracking']:
            config['tracking']['visualize_tracking_loss'] = False
        if "gaussian_distribution" not in config:
            config['gaussian_distribution'] = "isotropic"
        print(f"{config}")

        return config
    
    def update_prev_keyframes(self):
        """ update last stored keyframe indexs
    
        Attributes:
            prev_keyframe_idxs (List): last stored keyframe indexs
            
        """
        self.prev_keyframe_idxs = self.keyframe_time_indices.copy()

    def get_new_keyframe_idxs(self) -> torch.Tensor:
        """ get new keyframe indexs
    
        Returns:
            new_kf: [N], mask for new keyframes
        """
        prev_kf_idxs = torch.tensor(self.prev_keyframe_idxs)
        new_kf_idxs = torch.tensor(self.keyframe_time_indices)

        # Flatten each row into a single unique value (row-wise hashing)
        prev_flat = prev_kf_idxs.view(-1, 1)
        new_flat = new_kf_idxs.view(1, -1)

        # Find elements in new_kf_idxs that don't match any in prev_kf_idxs
        mask = (prev_flat == new_flat).any(dim=0)
        unique_new_kf_mask = ~mask

        return unique_new_kf_mask
