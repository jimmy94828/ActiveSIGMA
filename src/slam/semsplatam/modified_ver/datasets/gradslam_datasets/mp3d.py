import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted

from third_parties.splatam.datasets.gradslam_datasets.basedataset import GradSLAMDataset


class MP3DDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        load_semantics: Optional[bool] = False,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        self.load_semantics = load_semantics
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )
        if self.load_semantics:
            self.semantic_paths = self.get_semantic_filepaths()

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/results_habitat/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/results_habitat/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def get_semantic_filepaths(self):
        semantic_paths = []
        semantic_dir = os.path.join(self.input_folder, "results_habitat", "semantic")
        if self.load_semantics and os.path.isdir(semantic_dir):
            # Build a frame-index -> semantic file map. Handles names like:
            # semantic000123.npy, semantic_map_0123.npy, etc.
            semantic_by_idx = {}
            for spath in natsorted(glob.glob(f"{semantic_dir}/semantic*.npy")):
                stem = os.path.splitext(os.path.basename(spath))[0]
                digits = "".join([c for c in stem if c.isdigit()])
                if digits != "":
                    semantic_by_idx[int(digits)] = spath

            # Align semantic files to color paths so semantic indexing always matches dataset indexing
            for cpath in self.color_paths:
                fname = os.path.basename(cpath)
                num = "".join([c for c in fname if c.isdigit()])
                if num != "" and int(num) in semantic_by_idx:
                    semantic_paths.append(semantic_by_idx[int(num)])
                else:
                    semantic_paths.append(None)
        return semantic_paths


    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

    def read_semantic_from_file(self, semantic_file_path):
        semantic_map = np.load(semantic_file_path)
        semantic_map[semantic_map < 0] = 0
        return semantic_map

    def get_semantic_map(self,index):
        if index >= len(self.semantic_paths):
            h, w = self.desired_height, self.desired_width
            sem = torch.zeros((h, w), dtype=self.dtype)
            return sem.to(self.device),
        semantic_path = self.semantic_paths[index]
        if semantic_path is None or not os.path.exists(semantic_path):
            h, w = self.desired_height, self.desired_width
            sem = torch.zeros((h, w), dtype=self.dtype)
            return sem.to(self.device),
        semantics = self.read_semantic_from_file(semantic_path)
        if isinstance(semantics, np.ndarray):
            semantics = torch.from_numpy(semantics)
        semantics = semantics.to(dtype=self.dtype)
        # eval_helper expects target shape (H, W); resize if semantic map size differs.
        if semantics.ndim == 3 and semantics.shape[0] == 1:
            semantics = semantics[0]
        if semantics.ndim != 2:
            semantics = semantics.squeeze()
        if semantics.shape[0] != self.desired_height or semantics.shape[1] != self.desired_width:
            semantics = F.interpolate(
                semantics.unsqueeze(0).unsqueeze(0),
                size=(self.desired_height, self.desired_width),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        # if isinstance(semantics, np.ndarray):
        #     semantics = torch.from_numpy(semantics)
        return semantics.to(self.device).type(self.dtype),