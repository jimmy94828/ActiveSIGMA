import os
import matplotlib.cm as cm
from plyfile import PlyData, PlyElement
import torch
import numpy as np
from third_parties.splatam.utils.common_utils import params2cpu
from src.utils.general_utils import create_class_colormap,apply_colormap

def save_params_ckpt(output_params, output_variables, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)

    # also save semantic info
    for k in ['seman_cls_ids']:
        if isinstance(output_variables[k], torch.Tensor):
            to_save[k] = output_variables[k].detach().cpu().contiguous().numpy()
        else:
            to_save[k] = output_variables[k]

    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)

def save_rgb_ply(params,ckpt_output_dir, time_idx):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"rgb_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    rgbs = params['rgb_colors'].detach().cpu().contiguous().numpy()
    opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(rgbs, opacities, params, ply_savepath)

def save_semantic_ply(params,variables, ckpt_output_dir, time_idx, n_cls=150, colormap=None):
    os.makedirs(ckpt_output_dir, exist_ok=True)
    ply_name = f"semantic_GS_{time_idx:04}.ply"
    ply_savepath = os.path.join(ckpt_output_dir, ply_name)

    class_ids_indices = params['semantic_logits'].argmax(-1).cpu().numpy()
    topk_class = variables['seman_cls_ids'].cpu().numpy()
    class_ids = topk_class[np.arange(topk_class.shape[0]),class_ids_indices]
    if colormap == None:
        sem_colormap = create_class_colormap(n_cls)
    else:
        sem_colormap =  colormap
    sem_rgbs = apply_colormap(class_ids, sem_colormap) / 255.
    # sem_opacities = self.params['logit_opacities_seman'].detach().cpu().contiguous().numpy()
    sem_opacities = params['logit_opacities'].detach().cpu().contiguous().numpy()
    write_ply_file(sem_rgbs, sem_opacities, params, ply_savepath)

def write_ply_file(rgbs, opacities, params, ply_savepath):
    means = params['means3D'].detach().cpu().contiguous().numpy()
    rotations = params['unnorm_rotations'].detach().cpu().contiguous().numpy()
    scales = params['log_scales'].detach().cpu().contiguous().numpy()
    normals = np.zeros_like(means)
    C0 = 0.28209479177387814
    colors = (rgbs - 0.5) / C0
    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))
    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3', ]
    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_savepath)


def save_voxel_heat_ply(
    semantic_voxel_map,
    output_dir,
    ply_name="voxel_heat_final.ply",
    heat_min: float = 0.0,
    heat_max: float = 1.0,
    scalar_mode: str = "value",
    scalar_name: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    if scalar_mode == "value":
        scalar_fn = semantic_voxel_map.compute_voxel_value
        scalar_name = scalar_name or "heat"
    elif scalar_mode == "h_bar":
        scalar_fn = semantic_voxel_map.query_h_bar
        scalar_name = scalar_name or "h_bar"
    else:
        raise ValueError(f"Unsupported scalar_mode: {scalar_mode}")

    active_mask = getattr(semantic_voxel_map, 'alpha_v_active', None)
    if active_mask is not None:
        active_voxels = torch.where(active_mask)[0].tolist()
    else:
        active_voxels = sorted(getattr(semantic_voxel_map, 'directional_active_voxels', []))

    points = []
    heats = []
    for voxel_idx in active_voxels:
        heat_scalar = float(scalar_fn(int(voxel_idx)))
        if heat_scalar <= 0.0:
            continue
        center = semantic_voxel_map.get_voxel_center(int(voxel_idx)).detach().cpu().numpy()
        points.append(center.astype(np.float32))
        heats.append(heat_scalar)

    ply_savepath = os.path.join(output_dir, ply_name)
    if not points:
        empty_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), (scalar_name, 'f4')]
        PlyData([PlyElement.describe(np.empty(0, dtype=empty_dtype), 'vertex')]).write(ply_savepath)
        return

    points = np.stack(points, axis=0)
    heats = np.asarray(heats, dtype=np.float32)

    heat_span = float(heat_max) - float(heat_min)
    if heat_span <= 0.0:
        norm = np.zeros_like(heats, dtype=np.float32)
    else:
        norm = np.clip((heats - float(heat_min)) / heat_span, 0.0, 1.0)

    colors = cm.get_cmap('turbo')(norm)[..., :3]
    colors = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)

    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), (scalar_name, 'f4')]
    elements = np.empty(points.shape[0], dtype=dtype_full)
    elements['x'] = points[:, 0]
    elements['y'] = points[:, 1]
    elements['z'] = points[:, 2]
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    elements[scalar_name] = heats
    PlyData([PlyElement.describe(elements, 'vertex')]).write(ply_savepath)
