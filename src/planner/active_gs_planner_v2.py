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

from collections import defaultdict
import math
import mmengine
import numpy as np
import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import time

from src.planner.planner import compute_camera_pose
from src.planner.naruto_planner import NarutoPlanner
from src.planner.astar import path_planning as astar_planner
from src.slam.semsplatam.modified_ver.splatam.splatam import calc_shannon_entropy
from src.utils.general_utils import InfoPrinter
from src.planner.rotation_planning import rotation_planning
from src.planner.rrt_naruto import Node, is_collision_free
from src.data.pose_loader import PoseLoader
from src.planner.rotation_planner_v2 import smoothen_trajectory_v2 as smoothen_trajectory

from third_parties.splatam.utils.slam_external import calc_psnr
from third_parties.splatam.utils.common_utils import save_params


def remove_consecutive_duplicates(array_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Removes consecutive duplicate elements from a list of NumPy arrays.

    Args:
        array_list (list[np.ndarray]): A list of NumPy arrays, each with shape (3,).

    Returns:
        list[np.ndarray]: A new list with consecutive duplicate elements removed.
    """
    if not array_list:
        return []

    # Initialize the result list with the first element
    result = [array_list[0]]

    # Iterate through the list starting from the second element
    for i in range(1, len(array_list)):
        if not np.array_equal(array_list[i], array_list[i - 1]):
            result.append(array_list[i])

    return result


class ActiveGSPlannerv2(NarutoPlanner):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter,
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer

        Attributes:
            state (str): planner state
            is_observation_done (bool): is goal pose reached
            pose_loader (PoseLoader)
            obs_poses (List[torch.Tensor])
        """
        super(ActiveGSPlannerv2, self).__init__(main_cfg, info_printer)
        self.device = main_cfg.general.device
        
        ### initialize planner state ###
        self.state = "stay"

        ### initialize pose loader ###
        self.pose_loader = PoseLoader(main_cfg)

        self.obs_poses = []


        ### initialize refinement observation poses ###
        refine_steps = int(2 * math.pi * 0.3 / 0.1)
        self.refine_pose_set = self.generate_circular_trajectory(0.3, 0.1, refine_steps).numpy() # refine_pose-to-center 

        ### initialize explore and refine pool ###
        self.explore_pool = defaultdict(lambda: {})
        self._deleted_hot_keys = set()  # Track deleted HOT keys to prevent re-adding
        self._last_selected_explore_key = None
        self._last_ranked_explore_candidates = []
        self.refine_pool = {}
        self.planning_state = "exploration"
        self.exploration_stage = 0
        self.num_exploration_stage = self.planner_cfg.num_exploration_stage
        self.topk_cls_confidence = self.planner_cfg.topk_cls_confidence
        self.first_done_exploration = False

        ### initialize view direction sampling ###
        self.num_dir_samples = self.planner_cfg.num_dir_samples
        self.view_rot_samples = []
        self.view_rot_idx = []
        for i in range(self.num_exploration_stage):
            self.view_rot_samples.append(self.generate_rotation_samples(
                torch.from_numpy(self.planner_cfg.up_dir).to(self.device).float(),
                self.num_dir_samples[i],
            )) # 1, K, 4, 4
            self.view_rot_idx.append(torch.range(0, self.num_dir_samples[i]-1).unsqueeze(0).unsqueeze(2).to(self.device))

    def generate_circular_trajectory(self, radius: float, delta: float, steps: int) -> torch.Tensor:
        """
        Generate a circular trajectory around the current camera pose.
        Each step moves along the circle anticlockwise with a step size of delta.

        Args:
            radius (float): Radius of the circular trajectory on the XY plane.
            delta (float): Maximum step size for each movement.
            steps (int): Total number of steps for the circular trajectory.

        Returns:
            torch.Tensor: A tensor containing Nx4x4 matrices representing poses along the trajectory.
        """
        
        # Extract the initial rotation and translation from the pose
        current_pose = torch.eye(4)
        rotation = current_pose[:3, :3].clone()  # Copy of the initial rotation matrix
        initial_position = current_pose[:3, 3].clone()  # Initial position (x, y, z)
        
        # Calculate the initial point on the circle's surface (+X direction on XY plane)
        circle_start_position = initial_position.clone()
        circle_start_position[0] += radius  # Move along +X axis on XY plane


        # Initialize the list of poses with the starting pose on the circle
        init_path = self.interpolate_path(initial_position, circle_start_position, 0.1)
        trajectory = []
        for i in range(len(init_path)):
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation
            next_pose[:3, 3] = init_path[i]
            trajectory.append(next_pose)

        # Calculate the angle step based on `delta` and `radius`
        angle_step = delta / radius  # radians per step
        angle = 0.0  # Start angle at 0 (along +X)

        for _ in range(steps + 1):
            # Update angle for the next step
            angle += angle_step  # Move anticlockwise (positive angle in radians)

            # Compute the next position on the circle using polar coordinates (X = cos, Y = sin)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Calculate the new position by adding to the Z coordinate of the initial position
            next_position = initial_position.clone()
            next_position[0] += x
            next_position[1] += y

            # Create the 4x4 transformation matrix for this position
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation  # Keep the initial orientation constant
            next_pose[:3, 3] = next_position  # Update position

            # Append to the trajectory list
            trajectory.append(next_pose)

        # Initialize the list of poses with the starting pose on the circle
        last_path = self.interpolate_path(next_position, initial_position, 0.1)
        for i in range(len(last_path)):
            next_pose = torch.eye(4)
            next_pose[:3, :3] = rotation
            next_pose[:3, 3] = last_path[i]
            trajectory.append(next_pose)

        # Convert the list of poses to a tensor
        trajectory_tensor = torch.stack(trajectory)
        return trajectory_tensor

    def pose_conversion_sim2slam(self, sim_pose: torch.Tensor) -> torch.Tensor:
        """ Convert pose from Simulator system to SplaTAM system
        
        Args:
            sim_pose: Simulator pose
        
        Returns:
            slam_pose: SplaTAM pose
        """
        slam_pose = sim_pose.clone()
        slam_pose[:3, 1] *= -1
        slam_pose[:3, 2] *= -1
        slam_pose = self.sim2slam @ slam_pose
        return slam_pose

    def pose_conversion_slam2sim(self, slam_pose: torch.Tensor) -> torch.Tensor:
        """ Convert pose from SplaTAM system to Simulator system
        
        Args:
            slam_pose: SplaTAM pose
        
        Returns:
            sim_pose: Simulator pose
        """
        sim_pose = slam_pose.clone().to(self.sim2slam.device)
        sim_pose[:3, 1] *= -1
        sim_pose[:3, 2] *= -1
        sim_pose = torch.inverse(self.sim2slam) @ sim_pose
        return sim_pose
    
    def coord_conversion_slam2sim(self, slam_coords: torch.Tensor) -> torch.Tensor:
        """ Convert coordinate from SplaTAM system to Simulator system
        
        Args:
            slam_coords: [N,3], slam coordinates
        
        Returns:
            sim_coords: [N,3] Simulator pose
        """
        sim_coords = self.transform_points(torch.inverse(self.sim2slam), slam_coords)
        return sim_coords
    
    def coord_conversion_sim2slam(self, sim_coords: torch.Tensor) -> torch.Tensor:
        """ Convert coordinate from SplaTAM system to Simulator system
        
        Args:
            sim_coords: [N,3], slam coordinates
        
        Returns:
            slam_coords: [N,3] Simulator pose
        """
        slam_coords = self.transform_points(self.sim2slam, sim_coords)
        return slam_coords

    def init_data(self, 
                  sim2slam: torch.Tensor
                  ) -> None:
        """initialize data for naruto planner
    
        Args:
            # bbox (List, [3,2]): bounding box corners coordinates
    
        Attributes:
            gs_z_levels (List, [N])                  : Goal Space Z-levels. if not provided, unitformly samples from Z range.
            sim2slam (torch.Tensor): simator-to-slam conversion
            up_dir_slam (np.ndarray, [3]): up direction in SplaTAM system
            # voxel_size (float)                       : voxel size
            # bbox (np.ndarray, [3,2])                 : bounding box corners coordinates
            # Nx/Ny/Nz (int)                           : bounding box sizes
            # gs_x/y/z_range (torch.Tensor, [X/Y/Z])   : goal space X/Y/Z levels
            # goal_space_pts (torch.Tensor, [X*Y*Z, 3]): goal space candidate locations. Unit: voxel
        """
        # self.path = None
        # self.lookat_tgts = None

        ### load config data ###
        self.gs_z_levels = self.planner_cfg.get("gs_z_levels", -1)
        if self.gs_z_levels == -1:
            raise NotImplementedError
        self.sim2slam = sim2slam

        up_dir_sim = torch.from_numpy(self.planner_cfg.up_dir).float().to(self.sim2slam.device).unsqueeze(1)
        self.up_dir_slam = (self.sim2slam[:3, :3] @ up_dir_sim)[:, 0].cpu().numpy()


        ### bounding box ###
        self.bbox = np.asarray(self.main_cfg.slam.bbox_bound)
        self.voxel_size = self.main_cfg.slam.bbox_voxel_size
        self.discover_cls = []
        self.max_exploration_steps = self.planner_cfg.get("max_exploration_steps", 1000)

        # ## bounding box size (Unit: voxel) ##
        # Nx = round((bbox[0][1] - bbox[0][0]) / self.voxel_size + 0.0005) + 1
        # Ny = round((bbox[1][1] - bbox[1][0]) / self.voxel_size + 0.0005) + 1
        # Nz = round((bbox[2][1] - bbox[2][0]) / self.voxel_size + 0.0005) + 1
        # self.Nx = Nx
        # self.Ny = Ny
        # self.Nz = Nz

        # ### Goal Space ###
        # self.gs_x_range = torch.arange(0, self.Nx, 2)
        # self.gs_y_range = torch.arange(0, self.Ny, 2) 
        # self.gs_z_range = torch.arange(0, self.Nz, 2) 
        # self.gs_x, self.gs_y, self.gs_z = torch.meshgrid(self.gs_x_range, self.gs_y_range, self.gs_z_range, indexing="ij")
        # self.goal_space_pts = torch.cat([self.gs_x.reshape(-1, 1), 
        #                                  self.gs_y.reshape(-1, 1), 
        #                                  self.gs_z.reshape(-1, 1)], dim=1).cuda().float()

    def load_init_pose(self) -> torch.Tensor:
        """ load initial pose
    
        Returns:
            pose: [4, 4], camera-to-world pose
        """
        return self.pose_loader.load_init_pose().to(self.device)

    def check_observation_done(self):
        """ check if observation is completed at goal location 
        """
        is_obs_done = len(self.obs_poses) == 0
        return is_obs_done

    def rotation_planning_at_goal(self, 
                                   cur_pose : np.ndarray,
                                   goal_pose: np.ndarray
                                   ) -> np.ndarray:
        """ perform rotation planning
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            goal_pose (np.ndarray, [4,4]): goal pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system

        Attributes:
            rots (List): planned rotations. each element is (np.ndarray, [3,3])
        """
        rot = goal_pose[:3, :3]
        self.rots = rotation_planning(cur_pose[:3, :3], [rot], self.planner_cfg.max_rot_deg)

        new_pose = cur_pose.copy()
        return new_pose

    def rotating_at_goal(self, cur_pose: np.ndarray) -> np.ndarray:
        """ observing at the goal location using the observation poses in self.obs_poses. 
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        return self.rotating_at_current_loc(cur_pose)

    def observation_planning_at_goal(self, cur_pose : np.ndarray) -> np.ndarray:
        """ perform observation planning
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            goal_pose (np.ndarray, [4,4]): goal pose. Format: camera-to-world, RUB system
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system

        Attributes:
            obs_poses (List): planned observations. each element is (np.ndarray, [3,3])
        """
        ### FIXME: update observation planning ### 
        # refine_poses = cur_pose @ self.refine_pose_set
        # self.obs_poses = [i for i in refine_poses]
        # new_pose = cur_pose.copy()

        self.obs_poses = [cur_pose]
        new_pose = cur_pose.copy()
        return new_pose

    def observing_at_goal(self) -> np.ndarray:
        """ observing at the current location using the rotations in self.rots. 
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        new_pose = self.obs_poses.pop(0)
        return new_pose

    def update_state(self) -> None:
        """ update state machine for the planner
    
        Attributes:
            state (str): planner state
        """
        ##################################################
        ### Stay
        ##################################################
        if self.state == "stay":
            self.state = "planning"

        ##################################################
        ### planning
        ##################################################
        elif self.state == "planning":
            if self.planning_state == "post_refinement":
                self.state = "planning"
            else:
                ### FIXME: tmp apporach ###
                # self.state = "rotationPlanningAtStart"
                self.state = "movingToGoal"

        ##################################################
        ### rotation planning at start point
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            self.state = "rotatingAtStart"

        ##################################################
        ### rotating at start point
        ##################################################
        elif self.state == "rotatingAtStart":
            is_rotation_done = self.check_rotation_done()
            self.state = "movingToGoal" if is_rotation_done else "rotatingAtStart"

        ##################################################
        ### moving to goal
        ##################################################
        elif self.state == "movingToGoal":
            is_goal_reached = self.check_goal_reached()
            if is_goal_reached:
                # self.state = "rotationPlanningAtGoal"
                self.state = "planning"
            else:
                self.state = "movingToGoal"

        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            self.state = "rotatingAtGoal"

        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            is_rotation_done = self.check_rotation_done()
            if is_rotation_done:
                if self.planning_state == "refinement":
                    self.state = "observationPlanningAtGoal"
                else:
                    self.state = "planning"
            else:
                self.state = "rotatingAtGoal"

        ##################################################
        ### observation planning at goal
        ##################################################
        elif self.state == "observationPlanningAtGoal":
            self.state = "observingAtGoal"

        ##################################################
        ### observation at goal
        ##################################################
        elif self.state == "observingAtGoal":
            is_observation_done = self.check_observation_done()
            self.state = "planning" if is_observation_done else "observingAtGoal"


    def moving_to_goal(self, 
                       cur_pose  : np.ndarray,
                       lookat_loc: np.ndarray,
                       next_loc  : np.ndarray,
                       up_dir    : np.ndarray = None
                       ) -> np.ndarray:
        """ moving to goal while looking at lookat_loc
    
        Args:
            cur_pose (np.ndarray, [4,4]): current pose. Format: camera-to-world, RUB system
            lookat_loc (np.ndarray, [3]): look-at location
            next_loc (np.ndarray, [3]): next location
    
        Returns:
            new_pose (np.ndarray, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        rot = compute_camera_pose(next_loc, lookat_loc, up_dir=self.planner_cfg.up_dir if up_dir is None else up_dir, system="RDF")

        new_pose = cur_pose.copy()
        new_pose[:3, :3] = rot
        new_pose[:3, 3] = next_loc
        return new_pose

    def interpolate_path(self, A: torch.Tensor, B: torch.Tensor, step: float) -> List[torch.Tensor]:
        """
        Generates a list of intermediate points between two locations, A and B, with a given step size,
        excluding A and including B in the resulting list.

        Args:
            A (torch.Tensor): The starting location (1D tensor representing a point in n-dimensional space).
            B (torch.Tensor): The ending location (1D tensor representing a point in n-dimensional space).
            step (float): The step size between consecutive points.

        Returns:
            List[torch.Tensor]: A list of tensors representing intermediate points from A to B, excluding A and including B.
        
        Raises:
            ValueError: If A and B have different shapes or step size is non-positive.
        """
        if A.shape != B.shape:
            raise ValueError("A and B must have the same shape.")
        if step <= 0:
            raise ValueError("Step size must be a positive value.")

        ### Calculate the total distance between A and B ###
        distance = torch.norm(B - A)

        ### Calculate the number of intermediate steps needed ###
        num_steps = int(distance / step)

        ### Generate the intermediate points, excluding A and including B ###
        points = [A + (B - A) * ((i + 1) * step / distance) for i in range(num_steps)]

        ### Append B explicitly to ensure it is included ###
        if len(points) == 0 or ((points[-1] - B)!=0).any():
            points.append(B)
        
        return points

    def transform_points(self, transformation: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Transforms a set of 3D points by a 4x4 transformation matrix.

        Args:
            transformation (Tensor): A 4x4 tensor representing the transformation matrix.
            points (Tensor): An Nx3 tensor representing the set of 3D points.

        Returns:
            Tensor: An Nx3 tensor of transformed 3D points.
        """
        # Step 1: Convert points to homogeneous coordinates (Nx4)
        N = points.shape[0]
        ones = torch.ones((N, 1), dtype=points.dtype, device=points.device)  ### Add a column of ones
        points_homogeneous = torch.cat([points, ones], dim=1)  ### Concatenate to make Nx4 matrix

        # Step 2: Apply the transformation (4x4) to each point (Nx4)
        transformed_homogeneous = points_homogeneous @ transformation.T  ### Multiply by the transformation matrix
        
        # Step 3: Convert back from homogeneous to 3D by dividing by the last (homogeneous) coordinate
        transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3].unsqueeze(1)

        return transformed_points

    def convert_occ_grid_to_sdf(self, occ_grid):
        """
    
        Args:
            occ_grid: [D,H,W]
    
        Returns:
            
    
        Attributes:
            
        """
        sdf_vol = occ_grid.clone()
        sdf_vol[sdf_vol==1] = 0 # convert occupied to surface
        sdf_vol[sdf_vol<0] = 100 # convert free space to +ve
        # sdf_vol[sdf_vol==-1] = 100 # convert free space to +ve
        return sdf_vol

    def local_path_planning_rrt(self,
                         sdf_vol : np.ndarray,
                         cur_vxl: np.ndarray,
                         goal_vxl: np.ndarray
                         ) -> Tuple:
        """ Path planning
    
        Args:
            sdf_vol (np.ndarray, [X,Y,Z]): SDF volume
            cur_vxl (np.ndarray, [4,4]) : current vxl. 
            goal_vxl (np.ndarray, [3])   : goal location. Unit : voxel
    
        Returns:
            path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
        """
        # ### Force initial SDF to be empty space ###
        # if self.step == 0:
        #     sdf_vol = sdf_vol * 0. + 100.
        
        ## run local path planner ##
        self.local_planner.start_new_plan(
            start = cur_vxl,
            goal = goal_vxl,
            sdf_map = sdf_vol
        )
        target_reachable = self.local_planner.run(use_free_space=True)

        if not(target_reachable):
            partial_path = self._extract_rrt_partial_path(cur_vxl, goal_vxl)
            if partial_path is not None:
                self.info_printer(
                    "[Planner] RRT could not reach the target; using best reachable partial RRT path.",
                    self.step,
                    self.__class__.__name__,
                )
                return partial_path
            raise NotImplementedError

        ### find path ###
        path = self.local_planner.find_path()
        path = [i.get_xyz() for i in path[::-1]]
        path = torch.concat(path, dim=0)
        return path

    def _extract_rrt_partial_path(self, cur_vxl: np.ndarray, goal_vxl: np.ndarray):
        """Return a safe partial path from the already collision-checked RRT tree."""
        if not bool(self.planner_cfg.get('enable_rrt_partial_fallback', True)):
            return None

        nodes = getattr(self.local_planner, 'nodes', None)
        if nodes is None or len(nodes) <= 1:
            return None

        start = np.asarray(cur_vxl, dtype=np.float32)
        goal = np.asarray(goal_vxl, dtype=np.float32)
        total_goal_dist = float(np.linalg.norm(goal - start)) + 1e-8
        min_dist_m = float(self.planner_cfg.get('rrt_partial_min_dist', self.planner_cfg.trans_step_size))
        min_dist_vxl = max(min_dist_m / float(self.voxel_size), 1e-6)

        best_node = None
        best_score = -np.inf
        fallback_node = None
        fallback_dist = -np.inf
        for node in nodes[1:]:
            xyz = np.asarray(node._xyz_arr, dtype=np.float32)
            dist_from_start = float(np.linalg.norm(xyz - start))
            if dist_from_start < min_dist_vxl:
                continue

            if dist_from_start > fallback_dist:
                fallback_dist = dist_from_start
                fallback_node = node

            progress = (total_goal_dist - float(np.linalg.norm(goal - xyz))) / total_goal_dist
            score = progress + 0.05 * (dist_from_start / total_goal_dist)
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            best_node = fallback_node
        if best_node is None:
            return None

        path_nodes = []
        cur_node = best_node
        while cur_node is not None:
            path_nodes.append(cur_node.get_xyz())
            cur_node = cur_node.parent
        if len(path_nodes) <= 1:
            return None

        return torch.concat(path_nodes[::-1], dim=0)


    def path_planning(self, map: torch.Tensor, origin: torch.Tensor, start_pose: torch.Tensor, end_pose: torch.Tensor, trans_step: float, rot_step: float, voxel_size):
        """Plan translation path in occupancy space, then smooth orientation trajectory."""
        if torch.norm(start_pose[:3, 3] - end_pose[:3, 3]) < 1e-4:
            path = self.interpolate_path(start_pose[:3, 3], end_pose[:3, 3], self.planner_cfg.trans_step_size)
        else:
            # Transform start/end poses to simulator coordinates.
            start_pose_sim = self.pose_conversion_slam2sim(start_pose)
            end_pose_sim = self.pose_conversion_slam2sim(end_pose)

            # Convert to occupancy-grid local metric coordinates.
            start_loc = start_pose_sim[:3, 3] - origin
            end_loc = end_pose_sim[:3, 3] - origin

            # --- Sanity check: ensure end voxel is inside bounds and free. ---
            try:
                occ_cpu = map.detach().cpu()
            except Exception:
                occ_cpu = map

            def _voxel_is_free(world_offset: torch.Tensor) -> bool:
                """world_offset: (3,) sim coordinates relative to occ origin (meters)"""
                try:
                    vxl = (world_offset.detach().cpu() / self.voxel_size)
                    xi = int(torch.floor(vxl[0]).item())
                    yi = int(torch.floor(vxl[1]).item())
                    zi = int(torch.floor(vxl[2]).item())
                    Dx, Dy, Dz = occ_cpu.shape
                    if xi < 0 or xi >= Dx or yi < 0 or yi >= Dy or zi < 0 or zi >= Dz:
                        return False
                    return float(occ_cpu[xi, yi, zi].item()) == -1.0
                except Exception:
                    return False

            # Only ActiveSGM free-space voxels (-1.0) are valid RRT goals.
            if not _voxel_is_free(end_loc):
                self.info_printer(
                    "[Planner] Goal voxel is not known free space; trying next candidate.",
                    self.step,
                    self.__class__.__name__,
                )
                raise NotImplementedError("goal voxel is not known free space")

            # Local path planner (RRT).
            sdf_vol = self.convert_occ_grid_to_sdf(map)
            start_vxl = (start_loc / self.voxel_size).detach().cpu().numpy()
            start_idx = np.floor(start_vxl).astype(int)
            if np.all(start_idx >= 0) and np.all(start_idx < np.asarray(sdf_vol.shape)):
                # The simulator is already at the current camera pose; keep the RRT root traversable
                # even if the active ray update did not mark the exact sensor-origin voxel.
                sdf_vol[start_idx[0], start_idx[1], start_idx[2]] = 100
            path = self.local_path_planning_rrt(
                sdf_vol.detach().cpu().numpy(),
                start_vxl,
                (end_loc / self.voxel_size).detach().cpu().numpy(),
            )
            path *= self.voxel_size

            # Convert path back to SLAM coordinates.
            path += origin
            path = self.coord_conversion_sim2slam(path)
            path = [i for i in path[1:]]

        # Remove adjacent duplicate waypoints before orientation smoothing.
        path = [i.detach().cpu().numpy() for i in path]
        path = remove_consecutive_duplicates(path)

        start_pose_np = start_pose.detach().cpu().numpy()
        new_path = smoothen_trajectory(
            start_pose_np, 
            end_pose.detach().cpu().numpy(), 
            path, 
            rot_step, 
            -self.up_dir_slam
            )
        new_path = self._sanitize_pose_path([i for i in new_path], start_pose_np)
        return new_path

    def _sanitize_pose_path(self, poses, fallback_pose: np.ndarray):
        """Ensure planned poses are finite and have valid rotation matrices before simulation."""
        sanitized = []
        fixed_count = 0
        last_valid = np.asarray(fallback_pose, dtype=np.float32).copy()
        last_valid[3, :] = np.array([0, 0, 0, 1], dtype=np.float32)

        for pose in poses:
            arr = np.asarray(pose, dtype=np.float32).copy()
            if arr.shape != (4, 4):
                continue

            if not np.isfinite(arr[:3, 3]).all():
                arr[:3, 3] = last_valid[:3, 3]
                fixed_count += 1

            rot = arr[:3, :3]
            if not np.isfinite(rot).all():
                arr[:3, :3] = last_valid[:3, :3]
                fixed_count += 1
            else:
                try:
                    u, _, vh = np.linalg.svd(rot)
                    rot_ortho = u @ vh
                    if np.linalg.det(rot_ortho) < 0:
                        u[:, -1] *= -1
                        rot_ortho = u @ vh
                    if np.isfinite(rot_ortho).all():
                        arr[:3, :3] = rot_ortho.astype(np.float32)
                    else:
                        arr[:3, :3] = last_valid[:3, :3]
                        fixed_count += 1
                except np.linalg.LinAlgError:
                    arr[:3, :3] = last_valid[:3, :3]
                    fixed_count += 1

            arr[3, :] = np.array([0, 0, 0, 1], dtype=np.float32)
            if not np.isfinite(arr).all():
                arr = last_valid.copy()
                fixed_count += 1
            sanitized.append(arr)
            last_valid = arr

        if len(sanitized) == 0:
            sanitized.append(last_valid)
        if fixed_count > 0:
            self.info_printer(
                f"[Planner] Sanitized {fixed_count} invalid planned pose entries before simulation.",
                self.step,
                self.__class__.__name__,
            )
        return sanitized

    def compute_next_state_pose(self, 
                                cur_pose       : torch.Tensor,
                                gs_slam
                                ) -> torch.Tensor:
        """ compute next state pose
    
        Args:
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world (RUB; relative pose in SplaTAM)
            gs_slam: SplaTAM
    
        Returns:
            new_pose (torch.Tensor, [4,4]): new pose. Format: camera-to-world

        Attributes:
            traversability_mask (np.ndarray, [X,Y,Z]): valid goal space mask. get updated in self.uncertainty_aware_planning_v2()
            is_goal_reachable (bool)                 : is goal reachable
            lookat_tgts (List)                       : uncertaint target observation locations to lookat. each element is (np.ndarray, [3])
            path (List)                              : each element is a Node. [GoalNode, ..., CurrentNode]
        """
        ##################################################
        ### planning
        ##################################################
        if self.state == "planning":
            planner_state = f"{self.planning_state}_{self.exploration_stage}" if self.planning_state == "exploration" else self.planning_state
            self.timer.start(f"rendering_planning_{planner_state}", "Planner")
            planning_out = self.rendering_based_planning(cur_pose, gs_slam)
            self.timer.end(f"rendering_planning_{planner_state}")
            self.goal_pose = planning_out['new_pose'].clone() # RelPose c2w at SplaTAM
            ranked_candidates = planning_out.get('ranked_explore_candidates')
            if self.planning_state != "exploration" or not ranked_candidates:
                ranked_candidates = [(getattr(self, "_last_selected_explore_key", None), self.goal_pose)]
            first_selected_key = getattr(self, "_last_selected_explore_key", None)
            if self.planning_state == "exploration":
                rrt_fallback_topk = int(self.planner_cfg.get('rrt_fallback_topk', 5))
                if rrt_fallback_topk > 0:
                    ranked_candidates = ranked_candidates[:rrt_fallback_topk]

            if (self.goal_pose == cur_pose).all():
                self.path = [self.goal_pose]
            else:
                self.timer.start(f"rrt_planning_{planner_state}", "Planner")
                self.path = None
                for cand_key, cand_pose in ranked_candidates:
                    self.goal_pose = cand_pose.clone()
                    if (self.goal_pose == cur_pose).all():
                        self.path = [self.goal_pose]
                        break

                    try:
                        self.path = self.path_planning(
                            self.gs_slam.explr_map.occupancy_grid,
                            self.gs_slam.explr_map.origin,
                            cur_pose,
                            self.goal_pose,
                            self.planner_cfg.trans_step_size,
                            self.planner_cfg.rot_step_size,
                            self.main_cfg.slam.bbox_voxel_size,
                        )
                    except NotImplementedError as err:
                        self.info_printer(
                            f"[Planner] RRT failed for candidate {cand_key}: {err}; trying next ranked candidate.",
                            self.step,
                            self.__class__.__name__,
                        )
                        self._drop_failed_explore_candidate(cand_key)
                        continue

                    if cand_key is not None:
                        self._last_selected_explore_key = cand_key
                        if cand_key != first_selected_key and cand_key in self.explore_pool:
                            self.explore_pool[cand_key]['visit'] = self.explore_pool[cand_key].get('visit', 0) + 1
                    break

                if self.path is None:
                    self.info_printer(
                        "[Planner] No ranked exploration candidate is reachable by RRT in this cycle; requesting a fresh NBV evaluation.",
                        self.step,
                        self.__class__.__name__,
                    )
                    self.goal_pose = cur_pose.clone()
                    self.path = [self.goal_pose]
                self.timer.end(f"rrt_planning_{planner_state}")
            self.lookat_tgts = [self.goal_pose[:3, 3].detach().cpu().numpy()]
            new_pose = cur_pose

        ##################################################
        ### rotation planning at start location
        ##################################################
        elif self.state == "rotationPlanningAtStart":
            new_pose = self.rotation_planning_at_start(cur_pose.detach().cpu().numpy(), self.lookat_tgts[0], self.up_dir_slam, "RDF")

        ##################################################
        ### rotating at start location
        ##################################################
        elif self.state == "rotatingAtStart":
            new_pose = self.rotating_at_start(cur_pose.detach().cpu().numpy())

        ##################################################
        ### moving to goal 
        ##################################################
        elif self.state == "movingToGoal":
            # ### FIXME: add local path planner's path ###
            # next_loc = self.path[0].detach().cpu().numpy()
            # if len(self.path) == 1:
            #     new_pose = cur_pose.clone()
            #     new_pose[:3, 3] = self.goal_pose[:3, 3]
            # else:
            #     new_pose = self.moving_to_goal(cur_pose.detach().cpu().numpy(), self.lookat_tgts[0], next_loc, self.up_dir_slam)
            # self.path.pop(0)

            new_pose = self.path.pop(0)
            
        ##################################################
        ### rotation planning at goal location
        ##################################################
        elif self.state == "rotationPlanningAtGoal":
            new_pose = self.rotation_planning_at_goal(cur_pose.detach().cpu().numpy(), self.goal_pose.detach().cpu().numpy())

        ##################################################
        ### rotating at goal location
        ##################################################
        elif self.state == "rotatingAtGoal":
            new_pose = self.rotating_at_goal(cur_pose.detach().cpu().numpy())

        ##################################################
        ### observation planning at goal location
        ##################################################
        elif self.state == "observationPlanningAtGoal":
            new_pose = self.observation_planning_at_goal(cur_pose.detach().cpu().numpy())

        ##################################################
        ### observation at goal location
        ##################################################
        elif self.state == "observingAtGoal":
            new_pose = self.observing_at_goal()

        ##################################################
        ### Stay
        ##################################################
        elif self.state == "stay":
            new_pose = cur_pose.clone()

        return new_pose

    def generate_rotation_samples(self, 
                                  up: torch.Tensor,
                                  K: int = 10, 
                                  ) -> torch.Tensor:
        """
        Generates 1xKx4x4 rotation matrices with K evenly distributed viewing directions.

        Args:
            K (int): The number of viewing directions to generate.
            up (torch.Tensor): The up direction tensor (typically a unit vector).

        Returns:
            torch.Tensor: An 1xKx4x4 tensor where each 4x4 matrix represents a transformation for a viewing direction.
        
        """
        N = 1

        ### Fibonacci lattice parameters for regular sampling on a sphere ###
        golden_ratio = (1 + 5**0.5) / 2
        indices = torch.arange(0, K, dtype=torch.float32)

        ### Generate evenly spaced directions on the sphere ###
        theta = 2 * torch.pi * indices / golden_ratio
        z = 1 - (2 * indices + 1) / K
        radius = torch.sqrt(1 - z ** 2)
        directions = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=1).to(up.device)

        ### Initialize tensor to store 1xNx4x4 transformation matrices ###
        transformations = torch.zeros((N, K, 4, 4)).to(up.device)

        ### Compute right, true_up, and forward vectors in batches ###
        forward = -directions.unsqueeze(0).expand(-1, K, -1)  ### Shape NxKx3
        up_expanded = up.unsqueeze(0).unsqueeze(0)  ### Shape: (1, 1, 3)
        up_expanded = up_expanded.expand(forward.shape[0], forward.shape[1], -1)  ### Shape: (N, K, 3)
        right = torch.cross(up_expanded, forward, dim=2)
        right = right / right.norm(dim=2, keepdim=True)  ### Normalize right vector
        true_up = torch.cross(forward, right, dim=2)  ### Compute orthogonal up vector

        ### Stack right, true_up, and forward to form rotation part of the matrices ###
        rotation_matrices = torch.stack((right, true_up, forward), dim=3)  ### Shape NxKx3x3

        ### Create the 4x4 transformation matrices ###
        transformations[:, :, :3, :3] = rotation_matrices  ### Set rotation part
        
        ### Set the bottom row to [0, 0, 0, 1] for homogeneous coordinates ###
        transformations[:, :, 3, 3] = 1
        return transformations

    def generate_candidate_poses(self, 
                                points: torch.Tensor, 
                                transform_pose: torch.Tensor = None,
                                ) -> torch.Tensor:
        """
        Generates NxKx4x4 candidate poses for N points in space with K evenly distributed viewing directions per point.

        Args:
            points (torch.Tensor): A Nx3 tensor representing the locations of N points in space.
            transform_pose (torch.Tensor): 4x4 pose that transform all created poses

        Returns:
            torch.Tensor: An NxKx4x4 tensor where each 4x4 matrix represents a transformation for a viewing direction.
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input points tensor must be of shape Kx3.")

        N = points.shape[0]  ### Number of input points
        K = self.view_rot_samples[self.exploration_stage].shape[1]

        transformations = self.view_rot_samples[self.exploration_stage].repeat(N, 1, 1, 1)
        
        ### Create the 4x4 transformation matrices ###
        transformations[:, :, :3, 3] = points.unsqueeze(1).expand(-1, K, -1)  ### Set translation part
        
        # If transform_pose is provided, apply it to all generated poses
        if transform_pose is not None:
            # Ensure the transform_pose is a 4x4 tensor for matrix multiplication
            assert transform_pose.shape == (4, 4), "transform_pose must be a 4x4 tensor."
            # Apply the transformation to each pose using matrix multiplication
            transformations[:, :, :3, 1] *= -1
            transformations[:, :, :3, 2] *= -1 # RUB @ Sim
            transformations = transform_pose @ transformations

        return transformations

    def create_skybox_poses(self, 
                            points: torch.Tensor, 
                            up_direction: torch.Tensor, 
                            transform_pose: torch.Tensor = None
                            ) -> torch.Tensor:
        """
        Creates a set of poses for each point looking in 6 directions, like a skybox.

        Args:
            points (torch.Tensor): An Nx3 tensor representing the 3D positions.
            up_direction (torch.Tensor): A 3-element tensor representing the up direction of the world.
            transform_pose (torch.Tensor): 4x4 pose that transform all created poses

        Returns:
            torch.Tensor: An Nx6x4x4 tensor where each 4x4 matrix is a transformation matrix.
        """
        # Normalize the up direction
        up = F.normalize(up_direction, dim=0)  # Ensure it is a unit vector

        # Define the 6 directions to look at (forward, back, left, right, up, down)
        look_directions = torch.tensor([
            [1, 0, 0],   # +X
            [-1, 0, 0],  # -X
            [0, 1, 0],   # +Y
            [0, -1, 0],  # -Y
            [0, 0, 1],   # +Z
            [0, 0, -1]   # -Z
        ], dtype=points.dtype, device=points.device)  # 6x3 tensor

        # Prepare an output tensor (N x 6 x 4 x 4)
        N = points.shape[0]
        poses = torch.zeros((N, look_directions.shape[0], 4, 4), dtype=points.dtype, device=points.device)

        for i, direction in enumerate(look_directions):
            # view dir: -Z (backward)
            view_direction =  -F.normalize(direction, dim=0)  # Normalize the direction

            # Compute the right direction using cross product (up x view_dir)
            right_dir = torch.cross(up.expand_as(view_direction), view_direction)
            if all(right_dir == 0):
                right_dir[0] += 1.0
            else:
                right = F.normalize(right_dir, dim=0)

            # Compute the adjusted up direction using cross product (view_direction x right)
            adjusted_up = torch.cross(view_direction, right)

            # Create a rotation matrix (3x3)
            rotation_matrix = torch.stack([right, adjusted_up, view_direction], dim=1)  # 3x3 matrix

            # Create a transformation matrix (4x4) for each point
            poses[:, i, :3, :3] = rotation_matrix
            poses[:, i, :3, 3] = points
            poses[:, i, 3, 3] = 1.0

        # If transform_pose is provided, apply it to all skybox poses
        if transform_pose is not None:
            # Ensure the transform_pose is a 4x4 tensor for matrix multiplication
            assert transform_pose.shape == (4, 4), "transform_pose must be a 4x4 tensor."
            # Apply the transformation to each pose using matrix multiplication
            poses[:, :, :3, 1] *= -1
            poses[:, :, :3, 2] *= -1 # RUB @ Sim
            poses = transform_pose @ poses

        return poses

    def add_explore_pool_cand(self, cand_poses: torch.Tensor, cand_keys: torch.Tensor):
        """Add candidate poses to the exploration pool.
        
        HOT candidates (key[0]==-2) that have been deleted are prevented from
        being re-added to avoid infinite loops. Regular candidates can be re-added.
        """
        for i in range(cand_keys.shape[0]):
            key = tuple(cand_keys[i].cpu().numpy())
            # Skip re-adding deleted HOT keys (to prevent repeated cycles)
            if len(key) >= 1 and int(key[0]) == -2 and key in self._deleted_hot_keys:
                continue
            if key not in self.explore_pool:
                self.explore_pool[key]['pose'] = cand_poses[i]
                self.explore_pool[key]['visit'] = 0
                self.explore_pool[key]['pool_age'] = 0
    
    def update_explore_pool_cand(self, explore_igs: torch.Tensor, cand_keys: List[Tuple], next_visit: int):
        """Update IG, visit count, and pool age for exploration candidates."""
        for i in range(len(explore_igs)):
            self.explore_pool[cand_keys[i]]['ig'] = explore_igs[i]
            if 'visit' not in self.explore_pool[cand_keys[i]].keys():
                self.explore_pool[cand_keys[i]]['visit'] = 0
            if 'pool_age' not in self.explore_pool[cand_keys[i]].keys():
                self.explore_pool[cand_keys[i]]['pool_age'] = 0
            self.explore_pool[cand_keys[i]]['pool_age'] += 1

        self.explore_pool[cand_keys[next_visit]]['visit'] += 1


    def del_explore_pool_cand(self, explore_igs: torch.Tensor,
                              cand_keys: List[Tuple],
                              explore_thre: float,
                              recognize_thre: float):
        """Prune candidates with low geometry gain or excessive revisits."""
        explore_mask = explore_igs < (self.img_h * self.img_w) * explore_thre
        revisit = torch.tensor([self.explore_pool[cand]['visit'] for cand in cand_keys]).to(explore_mask.device)
        visit_mask = revisit > 3  # don't want to revisit a place by too many times
        rm_mask = explore_mask | visit_mask
        # Exploitation candidates (sentinel key: first element == -2) are near already-observed
        # semantic voxels so they naturally have lower geometry IG.
        # Keep them for a few rounds, then let geometry pruning apply to avoid pool explosion.
        hot_rm_by_explore_after_age = int(self.planner_cfg.get('hot_rm_by_explore_after_age', 2))
        for i, key in enumerate(cand_keys):
            if key[0] == -2:
                pool_age = int(self.explore_pool[key].get('pool_age', 0))
                if pool_age <= hot_rm_by_explore_after_age:
                    rm_mask[i] = visit_mask[i]
                else:
                    rm_mask[i] = explore_mask[i] | visit_mask[i]
        rm_idx = torch.where(rm_mask)[0]
        for i in rm_idx:
            key = cand_keys[int(i.item())]
            # Track deleted HOT keys to prevent re-adding
            if len(key) >= 1 and int(key[0]) == -2:
                self._deleted_hot_keys.add(key)
            del self.explore_pool[key]

    def del_low_nbv_pool_cand(self,
                              nbv_scores: torch.Tensor,
                              cand_keys:  List[Tuple],
                              nbv_thre:   float = 0.01,
                              pool_age_thre: int = 3,
                              ) -> int:
        """Prune low-score candidates that stayed in pool for enough rounds.

        Args:
            nbv_scores: [N], combined NBV score for each candidate (same order as cand_keys).
            cand_keys  : candidate key list with elements [X, Y, Z, R_i].
            nbv_thre   : absolute minimum acceptable NBV score.
            pool_age_thre: prune only if pool_age > pool_age_thre.

        Returns:
            int: number of candidates removed.
        """
        if nbv_scores.numel() == 0:
            return 0

        scores = nbv_scores.detach().float()
        low_abs = scores < float(nbv_thre)
        hot_pool_age_thre = int(self.planner_cfg.get('hot_nbv_pool_age_thre', 1))

        removed = 0
        for i in torch.where(low_abs)[0]:
            key = cand_keys[int(i.item())]
            if key not in self.explore_pool:
                continue
            pool_age = int(self.explore_pool[key].get('pool_age', 0))
            is_hot = len(key) >= 1 and int(key[0]) == -2
            age_thre = hot_pool_age_thre if is_hot else int(pool_age_thre)
            if pool_age <= age_thre:
                continue
            # Track deleted HOT keys to prevent re-adding
            if is_hot:
                self._deleted_hot_keys.add(key)
            del self.explore_pool[key]
            removed += 1

        if removed > 0:
            self.info_printer(
                f"[NBV Filter] Pruned {removed} candidates | pool_age>{pool_age_thre} and score<{nbv_thre:.4f}.",
                self.step,
                self.__class__.__name__,
            )
        return removed

    def _drop_failed_explore_candidate(self, cand_key) -> None:
        """Mark an RRT-unreachable exploration candidate and prune only after repeated failures."""
        if cand_key is None or cand_key not in self.explore_pool:
            return
        fail_count = int(self.explore_pool[cand_key].get('rrt_fail_count', 0)) + 1
        self.explore_pool[cand_key]['rrt_fail_count'] = fail_count
        prune_after = int(self.planner_cfg.get('rrt_fail_prune_after', 1))
        if prune_after <= 0 or fail_count < prune_after:
            return
        is_hot = len(cand_key) >= 1 and int(cand_key[0]) == -2
        if is_hot:
            self._deleted_hot_keys.add(cand_key)
        del self.explore_pool[cand_key]

    @staticmethod
    def _robust_normalize(x: torch.Tensor, low_pct: float = 10.0, high_pct: float = 90.0) -> torch.Tensor:
        """Robustly normalize a 1D score tensor into [0,1] using percentile clipping."""
        if x.numel() == 0:
            return x
        if x.numel() == 1:
            return torch.zeros_like(x)

        low_q = float(max(0.0, min(100.0, low_pct))) / 100.0
        high_q = float(max(0.0, min(100.0, high_pct))) / 100.0
        if high_q <= low_q:
            high_q = min(1.0, low_q + 0.1)

        lo = torch.quantile(x, low_q)
        hi = torch.quantile(x, high_q)
        span = (hi - lo).clamp(min=1e-8)
        return ((x - lo) / span).clamp(0.0, 1.0)

    def minmax_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalize a 1D score tensor into [0,1]."""
        if x.numel() == 0:
            return x
        if x.numel() == 1:
            return torch.zeros_like(x)

        lo = torch.min(x)
        hi = torch.max(x)
        span = (hi - lo).clamp(min=1e-8)
        return ((x - lo) / span).clamp(0.0, 1.0)
    
    def get_explore_pool_poses(self) -> Tuple[torch.Tensor, List[Tuple]]:
        """Return exploration pool poses and keys."""
        cand_poses = []
        cand_keys = []
        for key, val in self.explore_pool.items():
            cand_poses.append(val['pose'])
            cand_keys.append(key)
        cand_poses = torch.stack(cand_poses)
        return cand_poses, cand_keys

    def add_refine_pool_cand(self, kf_data: List):
        """ add keyframe candidates to refinement pool
    
        Args:
            kf_data: keyframe data
    
        Attributes:
            refine_pool: add candidates to refine_pool
            
        """
        for kf in kf_data:
            if self.step != 0:
                sim_c2w = self.pose_conversion_slam2sim(torch.inverse(kf['est_w2c']))
                kf_vxl = self.gs_slam.explr_map.transform_xyz_to_vxl(sim_c2w[:3, 3].unsqueeze(0))
                min_dist = self.gs_slam.explr_map.compute_min_distance_from_occ(
                    self.gs_slam.explr_map.occupancy_grid, 
                    kf_vxl
                    )[0]
                if min_dist * self.gs_slam.explr_map.voxel_size > self.planner_cfg.surface_dist_thre:
                    self.refine_pool[kf['id']] = kf    
            else:
                self.refine_pool[kf['id']] = kf
    
    def get_refine_pool_data(self) -> Tuple[List[Tuple], List[Tuple], torch.Tensor]:
        """ get refine pool data
    
        Returns:
            cand_data: refinement candidate data
            cand_keys: refinement candidate key
            cand_poses: refinement candidate poses
        """
        cand_data = []
        cand_keys = []
        cand_poses = []
        for key, val in self.refine_pool.items():
            cand_data.append(val)
            cand_keys.append(key)
            cand_poses.append(torch.inverse(val['est_w2c']))
        cand_poses = torch.stack(cand_poses)
        return cand_data, cand_keys, cand_poses

    def del_refine_pool_cand(self, 
                             color_igs: torch.Tensor, 
                             depth_igs: torch.Tensor,
                             seman_igs: torch.Tensor,
                             cand_keys: List[Tuple],
                             target_psnr: float = 34,
                             target_rel_depth_err: float = 0.01,
                             target_miou: float = 0.9,
                             ):
        """delete refine pool candidates that are already good enough
    
        Args:
            color_igs: [N], color information gain (PSNR)
            depth_igs: [N], depth informatoion gain (rel. depth error)
            cand_keys: candidate key with keyframe index
            target_psnr: target PSNR
            target_rel_depth_err: target relative depth error

        """
        rm_idx = torch.where((color_igs > target_psnr) * (depth_igs < target_rel_depth_err) * (seman_igs > target_miou))[0]
        for i in rm_idx:
            del self.refine_pool[cand_keys[i]]

    def rendering_based_planning(self, 
                                 cur_pose,
                                 gs_slam
                                 ):
        """ Rendering-based planning (goal searching)
    
        Args:
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world (RUB; relative pose in SplaTAM)
            gs_slam: SplaTAM
    
        Returns:
            Dict: planning output
                # - path (List)             : each element is a Node. [GoalNode, ..., CurrentNode]
    
        Attributes:
        """
        new_pose = cur_pose

        ##################################################
        ### Get Exploration map
        ##################################################
        # if self.planning_state in ["exploration", "refinement"]:
        if self.planning_state == "exploration" and self.exploration_stage < self.num_exploration_stage:
            self.info_printer(f"Current state: {self.state} | {self.planning_state}: Getting New Exploration Map", self.step, self.__class__.__name__)

            ### get exploration map (explored free space @ Sim coordinate system) ###
            gs_z_levels = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            new_free_voxels = gs_slam.explr_map.get_new_free_voxels(
                use_xyz_filter=True, 
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels
                )
            new_free_locs_sim = gs_slam.explr_map.origin + new_free_voxels * gs_slam.explr_map.voxel_size

            ### update previous_free_grid ###
            gs_z_levels = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            gs_slam.explr_map.update_prev_free_voxels(
                use_xyz_filter=True, 
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels
                )

            ##################################################
            ### sample new candidates
            ##################################################
            ### sample poses (SplaTAM system) from free space ###
            if new_free_locs_sim.shape[0] != 0:
                new_cand_poses = self.generate_candidate_poses(
                                        new_free_locs_sim, 
                                        gs_slam.explr_map.sim2slam
                                        ).reshape(-1, 4, 4)

                free_vxl_idx_exp = new_free_voxels.unsqueeze(1).repeat(1, self.num_dir_samples[self.exploration_stage], 1)
                view_rot_idx_exp = self.view_rot_idx[self.exploration_stage].repeat(new_free_voxels.shape[0], 1, 1)
                new_cand_pose_key = torch.cat([free_vxl_idx_exp, view_rot_idx_exp], dim=-1).reshape(-1, 4)

                self.add_explore_pool_cand(new_cand_poses, new_cand_pose_key)

            ##################################################
            ### Exploration
            ##################################################
            is_explore_done = len(self.explore_pool) == 0

            if self.step > self.max_exploration_steps:
                is_explore_done = True
                self.info_printer(
                    f"Current state: {self.state} | {self.planning_state}: Run out maximum exploration steps - {self.exploration_stage} , starting evaluation...",
                    self.step, self.__class__.__name__)
            
            ### FIXME: debug ###
            # is_explore_done = self.step > 50
            # if self.step > 50:
            #     self.gs_slam.print_and_save_result("exploration")
            
            ##################################################
            ### Evaluation when exploration is just done
            ##################################################

            if is_explore_done:
                if self.exploration_stage < self.num_exploration_stage:
                    self.info_printer(f"Current state: {self.state} | {self.planning_state}: Done Exploration Stage - {self.exploration_stage} , starting evaluation...", self.step, self.__class__.__name__)
                    eval_dir_suffix = f"exploration_stage_{self.exploration_stage}"
                    self.gs_slam.print_and_save_result(eval_dir_suffix, is_prune_gaussians=False, ignore_first_frame=True)
                    ### save step  ###
                    eval_dir = self.gs_slam.eval_dir + "_" + eval_dir_suffix
                    os.makedirs(eval_dir, exist_ok=True)
                    with open(os.path.join(eval_dir, "exploration_info.txt"), 'w') as f:
                        line = f"exploration_stage_{self.exploration_stage}_step: {self.step}\n"
                        f.writelines(line)

                        line = "global_keyframe: "
                        f.writelines(line)
                        
                        if self.main_cfg.slam.use_global_keyframe:
                            line = f"{sorted(self.gs_slam.global_keyframe_indices)}\n"
                            f.writelines(line)

                    self.exploration_stage += 1
                    self.planning_state = "exploration"
                
                if self.exploration_stage == self.num_exploration_stage:
                    self.info_printer(f"Current state: {self.state} | {self.planning_state}: Done All Exploration.", self.step, self.__class__.__name__)
                    if self.main_cfg.slam.use_global_keyframe:
                        self.planning_state = "post_refinement"
                        self.post_refine_counter = 0
                    else:
                        self.planning_state = "done"
                else:
                    ### reset exploration map ###
                    gs_z_levels = self.gs_z_levels[self.exploration_stage]
                    xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
                    gs_slam.explr_map.prev_free_voxels = torch.empty(0, 3).to(self.device)
                

            # if not(is_explore_done):
            else:
                self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Exploration Candidate I.G.", self.step, self.__class__.__name__)
                self.planning_state = "exploration"
                ### Get EXPLORE POOL poses and keys ###
                cand_poses, cand_keys = self.get_explore_pool_poses()
                explore_igs = []
                # recognize_igs = []
                gi_scores = []
                gd_scores = []
                voxel_mask = []

                ### compute distance between current pose and candidate poses ###
                dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6 # avoid zero dist case
                dists_sm = torch.nn.functional.softmax(dists, dim=0)
                for i, cand_pose in enumerate(cand_poses):
                    ### render data from candidate pose ###
                    img, depth, valid_mask, seen = gs_slam.render(cand_pose)
                    _, logits = gs_slam.render_semantic(cand_pose, seen)        # for the refinement stage

                    ##################################################
                    ### Ignore Simulation environement incomplete region
                    # FIXME: this simulation is time consuming.
                    # However, it is not related to our method but the imperfect simulation data.
                    ##################################################
                    depth_gt = self.sim.simulate(self.pose_conversion_slam2sim(cand_pose).detach().cpu().numpy(), no_print=True)['depth']
                    valid_sim_mask = depth_gt > 0.2 # 0.0 / 0.2 is the value that ignore rendering
                    valid_mask[0][~valid_sim_mask] = True

                    _, self.img_h, self.img_w = img.shape

                    ### compute EXPLORE I.G. ###
                    explore_ig = (valid_mask==0).sum()
                    explore_igs.append(explore_ig)

                    ### semantic GI / GD scores (if supported) ###
                    if hasattr(gs_slam, 'eval_candidate_semantic_gains'):
                        try:
                            GI, GD, pass_mask = gs_slam.eval_candidate_semantic_gains(cand_pose, seen=seen)
                        except Exception:
                            GI, GD, pass_mask = 0.0, 0.0, 0
                    else:
                        GI, GD, pass_mask = 0.0, 0.0, 0

                    gi_scores.append(float(GI))
                    gd_scores.append(float(GD))
                    voxel_mask.append(pass_mask)

                ### compute weighted exploration I.G., weighted by distance ###
                explore_igs = torch.stack(explore_igs).float()
                explore_igs_sm = torch.nn.functional.softmax(torch.log(explore_igs), dim=0)
                score_norm_low_pct = float(self.planner_cfg.get('score_norm_low_pct', 10.0))
                score_norm_high_pct = float(self.planner_cfg.get('score_norm_high_pct', 90.0))
                '''explore_igs_norm = self._robust_normalize(
                    explore_igs,
                    low_pct=score_norm_low_pct,
                    high_pct=score_norm_high_pct,
                )'''
                explore_igs_norm = self.minmax_normalize(explore_igs_sm)

                ### semantic GI / GD scores normalization and combination
                if len(gi_scores) > 0:
                    gi_scores = torch.tensor(gi_scores, device=dists.device).float()
                    gd_scores = torch.tensor(gd_scores, device=dists.device).float()
                else:
                    gi_scores = torch.zeros_like(dists)
                    gd_scores = torch.zeros_like(dists)

                '''if (gi_scores.max() - gi_scores.min()) > 1e-6:
                    gi_norm = self._robust_normalize(
                        gi_scores,
                        low_pct=score_norm_low_pct,
                        high_pct=score_norm_high_pct,
                    )
                else:
                    gi_norm = torch.zeros_like(gi_scores)'''
                    
                gi_norm = self.minmax_normalize(gi_scores)

                '''if (gd_scores.max() - gd_scores.min()) > 1e-6:
                    gd_norm = self._robust_normalize(
                        gd_scores,
                        low_pct=score_norm_low_pct,
                        high_pct=score_norm_high_pct,
                    )
                else:
                    gd_norm = torch.zeros_like(gd_scores)'''

                gd_norm = self.minmax_normalize(gd_scores)
                
                '''cost_norm = self._robust_normalize(
                    dists,
                    low_pct=score_norm_low_pct,
                    high_pct=score_norm_high_pct,
                )'''
                
                cost_norm = self.minmax_normalize(dists)

                # weights (fallback defaults)
                lambda_I = float(self.planner_cfg.get('lambda_I', 0.3))
                lambda_D = float(self.planner_cfg.get('lambda_D', 0.25))
                lambda_g = float(self.planner_cfg.get('lambda_g', 0.3))
                lambda_C = float(self.planner_cfg.get('lambda_C', 0.15))

                # final score combining geometry, semantic GI/GD and cost
                final_score = (
                    lambda_g * explore_igs_norm
                    + lambda_I * gi_norm
                    + lambda_D * gd_norm
                    - lambda_C * cost_norm
                )

                # original weighted explore score (kept for ablation), dists_sm use softmax
                seman_entropies = torch.zeros_like(final_score)
                weighted_explore_igs = (1 - dists_sm) * explore_igs_sm * seman_entropies

                # choose which scoring to use (default: semantic gains enabled)
                use_semantic_gains = bool(self.planner_cfg.get('use_semantic_gains', True))
                if use_semantic_gains:
                    next_visit = torch.argmax(final_score).item()
                    new_pose = cand_poses[next_visit]
                else:
                    next_visit = torch.argmax(weighted_explore_igs).item()
                    new_pose = cand_poses[next_visit]
                # print("Best pose px num: ", explore_igs[torch.argmax(weighted_explore_igs)])

                ### update explore pool ###
                self.update_explore_pool_cand(explore_igs, cand_keys, next_visit)

                ### remove explored view from EXPLORE POOL
                self.del_explore_pool_cand(explore_igs, cand_keys, self.planner_cfg.explore_thre,self.planner_cfg.recognize_thre)

                ### remove low-NBV score candidates from EXPLORE POOL ###
                nbv_score_thre = float(self.planner_cfg.get('nbv_score_thre', 0.01))  # TODO
                nbv_pool_age_thre = int(self.planner_cfg.get('nbv_pool_age_thre', 3))
                self.del_low_nbv_pool_cand(
                    final_score,
                    cand_keys,
                    nbv_score_thre,
                    pool_age_thre=nbv_pool_age_thre,
                )

                self.info_printer(f"Current state: {self.state} [Exploration Pool: {len(self.explore_pool)}]", self.step, self.__class__.__name__)
                self.info_printer(f"                            Exploration I.G.   : {explore_igs}", self.step, self.__class__.__name__)
                self.info_printer(f"                            Semantic Entropy   : {seman_entropies}", self.step,
                              self.__class__.__name__)
        ##################################################
        ### Refinement
        ##################################################
        if self.planning_state == "refinement":
            self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Refinement Candidate I.G.", self.step, self.__class__.__name__)
            self.planning_state = "refinement"
            ### get new keyframe and update prev keyframe###
            new_kfs = self.gs_slam.get_new_keyframe_idxs()
            self.gs_slam.update_prev_keyframes()

            ### update REFINE_POOL (add new keyframes to REFINE_POOL) ###
            selected_kf_list = [elem for elem, mask in zip(self.gs_slam.keyframe_list, new_kfs) if mask]
            self.add_refine_pool_cand(selected_kf_list)

            ### render poses in REFINE_POOL ###
            cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
            color_igs = []
            depth_igs = []
            seman_igs = []
            # TODO: add one more igs for miou

            ### compute distance between current pose and candidate poses ###
            dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6 # avoid zero dist case
            dists_sm = torch.nn.functional.softmax(dists, dim=0)
            for i, cand_pose in enumerate(cand_poses):
                color, depth, valid_mask,seen = gs_slam.render(cand_pose)
                cls_ids, logits = gs_slam.render_semantic(cand_pose, seen)

                ### compute REFINE I.G. ###
                valid_depth_mask = cand_data[i]['depth'] > 0
                color_ig = calc_psnr(color*valid_depth_mask, cand_data[i]['color']*valid_depth_mask).mean()
                color_igs.append(color_ig)
                depth_ig = (torch.abs(depth*valid_depth_mask - cand_data[i]['depth']*valid_depth_mask)/(cand_data[i]['depth']+1e-8)).sum() / valid_depth_mask.sum() 
                depth_igs.append(depth_ig)

                # TODO: add semantic ig
                seman_ig = calc_shannon_entropy(logits*valid_depth_mask,dim=0).mean()
                seman_igs.append(seman_ig)

            ### compute weighted Refinement I.G., weighted by distance ###
            color_igs = torch.stack(color_igs).float()
            depth_igs = torch.stack(depth_igs).float()
            seman_igs = torch.stack(seman_igs).float()

            color_igs_sm = 1 - torch.nn.functional.softmax(color_igs, dim=0)
            seman_igs_sm = torch.nn.functional.softmax(seman_igs, dim=0)
            refine_igs = color_igs_sm * seman_igs_sm
            best_key = torch.argmax(refine_igs)
            new_pose = cand_poses[best_key]

            ### remove refined views from REFINE_POOL ###
            self.del_refine_pool_cand(
                color_igs, 
                depth_igs,
                seman_igs,
                cand_keys,
                self.planner_cfg.color_ig_thre, 
                self.planner_cfg.depth_ig_thre,
                self.planner_cfg.seman_ig_thre,
                )
            self.info_printer(f"Refinement CandKey: {cand_keys}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement ColorIG: {color_igs}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement DepthIG: {depth_igs}", self.step, self.__class__.__name__)
            self.info_printer(f"Refinement best Cand: {cand_keys[best_key]} | {color_igs[best_key]} | {depth_igs[best_key]} | {refine_igs[best_key]}", self.step, self.__class__.__name__)
            self.info_printer(f"Current state: {self.state} [Refinement Pool: {len(self.refine_pool)}]", self.step, self.__class__.__name__)

            if len(self.refine_pool) == 0:
                self.planning_state = "post_refinement"

        ##################################################
        ### Post-Refinement
        ##################################################
        if self.planning_state == "post_refinement":
            self.post_refine_counter += 1
            if self.post_refine_counter >= self.planner_cfg.post_refine_steps:
                self.planning_state = "done"
            if self.step % self.planner_cfg.post_refinement_eval_freq == 0:


                self.info_printer(f"Current state: {self.state} | {self.planning_state}: Evaluate Post-Refinement Candidate I.G.", self.step, self.__class__.__name__)
                self.planning_state = "post_refinement"
                ### get new keyframe and update prev keyframe###
                # new_kfs = self.gs_slam.get_new_keyframe_idxs()
                # self.gs_slam.update_prev_keyframes()

                ### update REFINE_POOL (add new keyframes to REFINE_POOL) ###
                ### FIXME: only use global keyframe ###
                # selected_kf_list = [elem for elem, mask in zip(self.gs_slam.keyframe_list, new_kfs) if mask]
                if len(self.refine_pool) == 1:
                    selected_kf_list = [self.gs_slam.keyframe_list[i] for i in self.gs_slam.global_keyframe_indices]
                    self.add_refine_pool_cand(selected_kf_list)

                ### render poses in REFINE_POOL ###
                cand_data, cand_keys, cand_poses = self.get_refine_pool_data()
                color_igs = []
                seman_igs = []

                ### compute distance between current pose and candidate poses ###
                for i, cand_pose in enumerate(cand_poses):
                    color, depth, valid_mask, seen = gs_slam.render(cand_pose)
                    pred_cls_ids, pred_logits = gs_slam.render_semantic(cand_pose, seen)

                    _, img_h, img_w = color.shape

                    ### compute REFINE I.G. ###
                    valid_depth_mask = cand_data[i]['depth'] > 0
                    color_ig = calc_psnr(color*valid_depth_mask, cand_data[i]['color']*valid_depth_mask).mean()
                    color_igs.append(color_ig)

                    topk_probs, _ = torch.topk(pred_logits, k=16,dim=0)
                    recognize_mask = topk_probs.sum(0)>0.5
                    seman_ig = (recognize_mask & valid_depth_mask).sum()/valid_depth_mask.sum()

                    seman_igs.append(seman_ig)

                ### compute weighted Refinement I.G., weighted by distance ###
                color_igs = torch.stack(color_igs).float()
                seman_igs = torch.stack(seman_igs).float()

                ### remove refined views from REFINE_POOL ###
                self.info_printer(f"Current state: {self.state} [Refinement Pool: {len(self.refine_pool)}]", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement ColorIG: {color_igs}", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement ColorIG [Min, Avg]: [{torch.min(color_igs).item():.2f}, {torch.mean(color_igs).item():.2f}]", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement SemanticIG: {seman_igs}", self.step, self.__class__.__name__)
                self.info_printer(f"Refinement SemanticIG [Min, Avg]: [{torch.min(seman_igs).item():.2f}, {torch.mean(seman_igs).item():.2f}]",
                    self.step, self.__class__.__name__)
                # if len(self.refine_pool) == 0:
                ### When 90% of global keyframes are good, then it is done ###
                color_requirement = (color_igs > self.main_cfg.slam.global_keyframe.color_thre).sum()/len(color_igs) > 0.9
                seman_requirement = (seman_igs > self.main_cfg.slam.global_keyframe.seman_thre).sum()/len(seman_igs) > 0.9



                if color_requirement & seman_requirement:
                    self.planning_state = "done"

        if self.planning_state == "done":
            self.planning_state = "done"
            self.info_printer(f"Current state: Exploration + Refinement All Done!", self.step, self.__class__.__name__)
    
        out = dict(
            new_pose = new_pose,
        )
        return out

    def main(self, 
             cur_pose       : torch.Tensor,
             gs_slam,
             ) -> torch.Tensor:
        """ Naruto Planner main function
    
        Args:
            gs_slam: 
            cur_pose (torch.Tensor, [4,4]): current pose. Format: camera-to-world, RUB system
            is_new_vols (bool)          : is uncert_sdf_vols new optimized volumes
    
        Returns:
            new_pose (torch.Tensor, [4,4]): new pose. Format: camera-to-world, RUB system
        """
        self.gs_slam = gs_slam
        self.update_state()
        self.info_printer(f"Current state: {self.state}", self.step, self.__class__.__name__)
        new_pose = self.compute_next_state_pose(cur_pose, gs_slam)
        if type(new_pose) == np.ndarray:
            new_pose = torch.from_numpy(new_pose).float().to(cur_pose.device)
        return new_pose
