#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    split_manifest: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _sparse_split_requested(sparse_train_images="", sparse_train_indices="", sparse_train_count=0):
    return bool(sparse_train_images) or bool(sparse_train_indices) or sparse_train_count > 0


def _read_colmap_extrinsics_intrinsics(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        sparse_format = "bin"
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        sparse_format = "txt"

    return cam_extrinsics, cam_intrinsics, sparse_format


def _read_name_list_txt(path):
    if path is None or path == "":
        return []
    with open(path, "r", encoding="utf-8") as f:
        names = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(os.path.basename(line.replace("\\", "/")))
    return names


def _camera_name_keys(image_name):
    normalized = os.path.basename(str(image_name).strip().replace("\\", "/"))
    stem = Path(normalized).stem
    return {normalized, normalized.lower(), stem, stem.lower()}


def _camera_name_key_set(cam_infos):
    keys = set()
    for cam_info in cam_infos:
        keys.update(_camera_name_keys(cam_info.image_name))
    return keys


def _read_sparse_train_image_names(value, scene_path):
    if not value:
        return []

    candidate_paths = [value]
    if not os.path.isabs(value):
        candidate_paths.extend([
            os.path.join(scene_path, value),
            os.path.join(scene_path, "sparse/0", value),
        ])

    list_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
    if list_path is None:
        return [token.strip() for token in value.split(",") if token.strip()]

    if list_path.lower().endswith(".json"):
        with open(list_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            for key in ("train", "train_images", "images", "frames"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Sparse train image JSON must contain a list: {list_path}")

        names = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("image_name") or item.get("file_name") or item.get("file_path") or item.get("path")
            else:
                name = item
            if name:
                names.append(str(name).strip())
        return names

    with open(list_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]


def _parse_sparse_train_indices(value, num_cameras):
    if not value:
        return []

    indices = []
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            continue

        if ":" in token:
            parts = token.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid sparse train index range: {token}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            indices.extend(range(start, stop, step))
        elif "-" in token and not token.startswith("-"):
            start, stop = [int(v) for v in token.split("-", 1)]
            step = 1 if stop >= start else -1
            indices.extend(range(start, stop + step, step))
        else:
            indices.append(int(token))

    normalized = []
    seen = set()
    for idx in indices:
        if idx < 0:
            idx = num_cameras + idx
        if idx < 0 or idx >= num_cameras:
            raise IndexError(f"Sparse train index {idx} is out of range for {num_cameras} cameras")
        if idx not in seen:
            normalized.append(idx)
            seen.add(idx)
    return normalized


def _evenly_spaced_sparse_indices(num_cameras, sparse_train_count):
    if sparse_train_count <= 0:
        return []
    if sparse_train_count > num_cameras:
        raise ValueError(
            f"sparse_train_count={sparse_train_count} is larger than the number of cameras ({num_cameras})"
        )
    return np.linspace(0, num_cameras - 1, sparse_train_count, dtype=int).tolist()


def _build_sparse_train_test_split(all_cam_infos, scene_path, sparse_train_images="", sparse_train_indices="", sparse_train_count=0):
    selected_train = []
    selected_names = set()

    def add_train_camera(cam_info):
        if cam_info.image_name not in selected_names:
            selected_train.append(cam_info)
            selected_names.add(cam_info.image_name)

    sparse_names = _read_sparse_train_image_names(sparse_train_images, scene_path)
    if sparse_names:
        unmatched = []
        for name in sparse_names:
            name_keys = _camera_name_keys(name)
            matches = [cam for cam in all_cam_infos if _camera_name_keys(cam.image_name) & name_keys]
            if not matches:
                unmatched.append(name)
            for cam in matches:
                add_train_camera(cam)
        if unmatched:
            print("[WARN] Sparse train image names not found: {}".format(", ".join(unmatched)))
    elif sparse_train_indices:
        for idx in _parse_sparse_train_indices(sparse_train_indices, len(all_cam_infos)):
            add_train_camera(all_cam_infos[idx])
    elif sparse_train_count > 0:
        for idx in _evenly_spaced_sparse_indices(len(all_cam_infos), sparse_train_count):
            add_train_camera(all_cam_infos[idx])

    if not selected_train:
        raise ValueError("Sparse-view split was requested, but no train cameras were selected.")

    train_name_set = {cam.image_name for cam in selected_train}
    train_cam_infos = [cam._replace(is_test=False) for cam in selected_train]
    test_cam_infos = [cam._replace(is_test=True) for cam in all_cam_infos if cam.image_name not in train_name_set]

    print(
        "[SPARSE-VIEW SPLIT] full={} train={} test={} (test = full - train)".format(
            len(all_cam_infos), len(train_cam_infos), len(test_cam_infos)
        )
    )
    if len(test_cam_infos) == 0:
        print("[WARN] Sparse-view split produced an empty test set.")

    return train_cam_infos, test_cam_infos


def _build_full_source_test_split(train_cam_infos, full_cam_infos):
    train_name_keys = _camera_name_key_set(train_cam_infos)
    test_cam_infos = [
        cam._replace(is_test=True)
        for cam in full_cam_infos
        if not (_camera_name_keys(cam.image_name) & train_name_keys)
    ]

    print(
        "[FULL TEST SPLIT] train={} full_test_source={} test={} (test = full_test_source - train names)".format(
            len(train_cam_infos), len(full_cam_infos), len(test_cam_infos)
        )
    )
    if len(test_cam_infos) == 0:
        print("[WARN] Full-test split produced an empty test set.")

    return test_cam_infos


def _select_eval_test_names(eval_source_path, eval_extrinsics, split_mode, llffhold, test_view_list_path):
    all_names = sorted([eval_extrinsics[k].name for k in eval_extrinsics])

    if split_mode == "llffhold":
        if llffhold <= 0:
            raise ValueError("--dpcr_eval_llffhold must be > 0 when split_mode='llffhold'")
        test_names = [name for idx, name in enumerate(all_names) if idx % llffhold == 0]
    elif split_mode == "test_txt":
        test_txt = os.path.join(eval_source_path, "sparse/0", "test.txt")
        if not os.path.exists(test_txt):
            raise FileNotFoundError(f"Expected test split file not found: {test_txt}")
        test_names = _read_name_list_txt(test_txt)
    elif split_mode == "manifest":
        if not test_view_list_path:
            raise ValueError("--dpcr_eval_test_view_list is required when split_mode='manifest'")
        test_names = _read_name_list_txt(test_view_list_path)
    else:
        raise ValueError(f"Unknown --dpcr_eval_split_mode: {split_mode}")

    all_name_set = {os.path.basename(n) for n in all_names}
    missing = sorted(set(os.path.basename(n) for n in test_names) - all_name_set)
    if missing:
        raise ValueError(
            f"Some requested eval test images are not present in eval source COLMAP images: {missing[:20]}"
        )

    return [os.path.basename(n) for n in test_names], [os.path.basename(n) for n in all_names]


def _filter_cam_infos_by_names(cam_infos, include_names=None, exclude_names=None):
    include_set = set(os.path.basename(n) for n in include_names) if include_names else None
    exclude_set = set(os.path.basename(n) for n in exclude_names) if exclude_names else set()

    out = []
    for c in cam_infos:
        name = os.path.basename(c.image_name)
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        out.append(c)
    return out


def _camera_center_from_caminfo(cam):
    return -cam.R @ cam.T


def _check_same_frame_by_common_cameras(train_cam_infos, eval_cam_infos, min_common=4, tol=1e-3):
    train_by_name = {os.path.basename(c.image_name): c for c in train_cam_infos}
    eval_by_name = {os.path.basename(c.image_name): c for c in eval_cam_infos}

    common = sorted(set(train_by_name.keys()) & set(eval_by_name.keys()))

    if len(common) < min_common:
        raise ValueError(
            f"Not enough common camera names to verify coordinate frame. "
            f"common={len(common)}, required={min_common}. "
            "Use --dpcr_eval_frame_mode skip only if you are absolutely sure both folders share the same coordinate frame."
        )

    train_centers = np.stack([_camera_center_from_caminfo(train_by_name[n]) for n in common], axis=0)
    eval_centers = np.stack([_camera_center_from_caminfo(eval_by_name[n]) for n in common], axis=0)

    scale_ref = max(np.linalg.norm(train_centers.max(axis=0) - train_centers.min(axis=0)), 1e-8)
    rmse = np.sqrt(np.mean(np.sum((train_centers - eval_centers) ** 2, axis=1)))
    rel_rmse = rmse / scale_ref

    if rel_rmse > tol:
        raise ValueError(
            "Sparse train source and eval source do not appear to share the same coordinate frame.\n"
            f"common_images={len(common)}, center_rmse={rmse}, relative_rmse={rel_rmse}, tol={tol}\n"
            "This usually happens when the 12-view folder was reconstructed separately by VGGT/SfM. "
            "Use --dpcr_eval_frame_mode align_umeyama, or regenerate the sparse folder by subsetting cameras from the full source coordinate frame."
        )

    return {
        "common_count": len(common),
        "center_rmse": float(rmse),
        "relative_rmse": float(rel_rmse),
        "mode": "strict",
    }


def _umeyama_similarity(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    assert src.shape == dst.shape
    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for Umeyama alignment.")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1

    R = U @ D @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c ** 2, axis=1))
        scale = np.trace(np.diag(S) @ D) / max(var_src, 1e-12)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    return float(scale), R.astype(np.float64), t.astype(np.float64)


def _transform_camera_info_sim3(cam, scale, R_align, t_align):
    Rwc_old = cam.R
    tcw_old = cam.T

    C_old = -Rwc_old @ tcw_old
    C_new = scale * (R_align @ C_old) + t_align
    Rwc_new = R_align @ Rwc_old
    Rcw_new = Rwc_new.T
    tcw_new = -Rcw_new @ C_new

    return CameraInfo(
        uid=cam.uid,
        R=Rwc_new.astype(np.float32),
        T=tcw_new.astype(np.float32),
        FovY=cam.FovY,
        FovX=cam.FovX,
        depth_params=cam.depth_params,
        image_path=cam.image_path,
        image_name=cam.image_name,
        depth_path=cam.depth_path,
        width=cam.width,
        height=cam.height,
        is_test=cam.is_test,
    )


def _align_eval_cameras_to_train_frame(train_cam_infos, eval_all_cam_infos, test_cam_infos, min_common=4):
    train_by_name = {os.path.basename(c.image_name): c for c in train_cam_infos}
    eval_by_name = {os.path.basename(c.image_name): c for c in eval_all_cam_infos}

    common = sorted(set(train_by_name.keys()) & set(eval_by_name.keys()))

    if len(common) < min_common:
        raise ValueError(
            f"Not enough common images for Umeyama alignment: common={len(common)}, required={min_common}. "
            "Cannot align full eval cameras to sparse train frame."
        )

    src_eval = np.stack([_camera_center_from_caminfo(eval_by_name[n]) for n in common], axis=0)
    dst_train = np.stack([_camera_center_from_caminfo(train_by_name[n]) for n in common], axis=0)

    scale, R_align, t_align = _umeyama_similarity(src_eval, dst_train, with_scale=True)

    aligned_test = [
        _transform_camera_info_sim3(c, scale, R_align, t_align)
        for c in test_cam_infos
    ]

    aligned_eval_common = [
        _transform_camera_info_sim3(eval_by_name[n], scale, R_align, t_align)
        for n in common
    ]

    aligned_by_name = {os.path.basename(c.image_name): c for c in aligned_eval_common}

    errors = []
    for n in common:
        c_train = _camera_center_from_caminfo(train_by_name[n])
        c_eval_aligned = _camera_center_from_caminfo(aligned_by_name[n])
        errors.append(np.linalg.norm(c_train - c_eval_aligned))

    rmse = float(np.sqrt(np.mean(np.square(errors))))

    return aligned_test, {
        "mode": "align_umeyama",
        "common_count": len(common),
        "scale": float(scale),
        "R": R_align.tolist(),
        "t": t_align.tolist(),
        "alignment_center_rmse": rmse,
        "common_names": common,
    }


def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    test_cam_name_set = set(os.path.basename(name) for name in test_cam_names_list)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=os.path.basename(image_name) in test_cam_name_set)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    llffhold=8,
    sparse_train_images="",
    sparse_train_indices="",
    sparse_train_count=0,
    full_test_source_path="",
    full_test_images="",
    dpcr_eval_source_path="",
    dpcr_eval_images="",
    dpcr_eval_split_mode="llffhold",
    dpcr_eval_llffhold=8,
    dpcr_train_view_list="",
    dpcr_eval_test_view_list="",
    dpcr_eval_require_disjoint=True,
    dpcr_eval_frame_mode="strict",
    dpcr_eval_alignment_min_common=4,
    dpcr_eval_frame_check_tol=1e-3,
):
    train_source_path = path
    cam_extrinsics, cam_intrinsics, train_sparse_format = _read_colmap_extrinsics_intrinsics(train_source_path)

    depth_params_file = os.path.join(train_source_path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    sparse_split_active = _sparse_split_requested(
        sparse_train_images=sparse_train_images,
        sparse_train_indices=sparse_train_indices,
        sparse_train_count=sparse_train_count,
    )
    use_external_eval = bool(eval and dpcr_eval_source_path)
    full_test_split_active = bool(full_test_source_path) and not use_external_eval
    split_manifest = {}

    if use_external_eval:
        eval_extrinsics, eval_intrinsics, eval_sparse_format = _read_colmap_extrinsics_intrinsics(dpcr_eval_source_path)
        test_names, eval_all_names = _select_eval_test_names(
            eval_source_path=dpcr_eval_source_path,
            eval_extrinsics=eval_extrinsics,
            split_mode=dpcr_eval_split_mode,
            llffhold=dpcr_eval_llffhold,
            test_view_list_path=dpcr_eval_test_view_list,
        )

        all_sparse_train_names = sorted([os.path.basename(cam_extrinsics[k].name) for k in cam_extrinsics])
        if dpcr_train_view_list:
            train_names = _read_name_list_txt(dpcr_train_view_list)
        else:
            train_names = all_sparse_train_names
        train_names = list(dict.fromkeys(os.path.basename(n) for n in train_names))

        missing_train = sorted(set(train_names) - set(all_sparse_train_names))
        if missing_train:
            raise ValueError(f"Train view list contains images not found in sparse train source: {missing_train[:20]}")

        overlap = sorted(set(train_names) & set(test_names))
        if overlap and dpcr_eval_require_disjoint:
            raise ValueError(
                "Train/test leakage detected. These images are in sparse train source but also in fixed eval test split: "
                f"{overlap[:50]}\n"
                "Regenerate the sparse-view folder using only non-test images from the full dataset, "
                "or add --no_dpcr_eval_require_disjoint only for debugging."
            )

        train_reading_dir = "images" if images is None else images
        train_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            depths_params=depths_params,
            images_folder=os.path.join(train_source_path, train_reading_dir),
            depths_folder=os.path.join(train_source_path, depths) if depths != "" else "",
            test_cam_names_list=[],
        )
        train_cam_infos = sorted(
            _filter_cam_infos_by_names(train_cam_infos_unsorted, include_names=train_names),
            key=lambda x: x.image_name
        )

        eval_reading_dir = dpcr_eval_images if dpcr_eval_images else train_reading_dir
        eval_cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=eval_extrinsics,
            cam_intrinsics=eval_intrinsics,
            depths_params=None,
            images_folder=os.path.join(dpcr_eval_source_path, eval_reading_dir),
            depths_folder="",
            test_cam_names_list=test_names,
        )
        eval_all_cam_infos = sorted(eval_cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        test_cam_infos = sorted(
            _filter_cam_infos_by_names(eval_all_cam_infos, include_names=test_names),
            key=lambda x: x.image_name
        )

        if dpcr_eval_frame_mode == "strict":
            frame_report = _check_same_frame_by_common_cameras(
                train_cam_infos=train_cam_infos,
                eval_cam_infos=eval_all_cam_infos,
                min_common=dpcr_eval_alignment_min_common,
                tol=dpcr_eval_frame_check_tol,
            )
        elif dpcr_eval_frame_mode == "align_umeyama":
            test_cam_infos, frame_report = _align_eval_cameras_to_train_frame(
                train_cam_infos=train_cam_infos,
                eval_all_cam_infos=eval_all_cam_infos,
                test_cam_infos=test_cam_infos,
                min_common=dpcr_eval_alignment_min_common,
            )
        elif dpcr_eval_frame_mode == "skip":
            frame_report = {"mode": "skip", "warning": "coordinate frame check skipped"}
        else:
            raise ValueError(f"Unknown --dpcr_eval_frame_mode: {dpcr_eval_frame_mode}")

        eval_test_set = set(test_names)
        train_set = set(train_names)
        eval_all_set = set(eval_all_names)
        unused_names = sorted(eval_all_set - eval_test_set - train_set)

        split_manifest = {
            "protocol": "dpcr_sparse_train_external_eval",
            "train_source_path": os.path.abspath(train_source_path),
            "eval_source_path": os.path.abspath(dpcr_eval_source_path),
            "train_sparse_format": train_sparse_format,
            "eval_sparse_format": eval_sparse_format,
            "eval_split_mode": dpcr_eval_split_mode,
            "eval_llffhold": int(dpcr_eval_llffhold),
            "sort_key": "sorted image_name",
            "train_count": len(train_cam_infos),
            "test_count": len(test_cam_infos),
            "eval_full_count": len(eval_all_names),
            "unused_count": len(unused_names),
            "train_image_names": sorted([os.path.basename(c.image_name) for c in train_cam_infos]),
            "test_image_names": sorted([os.path.basename(c.image_name) for c in test_cam_infos]),
            "unused_image_names": unused_names,
            "overlap_train_test": sorted(train_set & eval_test_set),
            "frame_report": frame_report,
        }

        print("------------DPCR EXTERNAL EVAL SPLIT-------------")
        print(f"[SPLIT] train_source: {train_source_path}")
        print(f"[SPLIT] eval_source : {dpcr_eval_source_path}")
        print(f"[SPLIT] train_count : {len(train_cam_infos)}")
        print(f"[SPLIT] test_count  : {len(test_cam_infos)}")
        print(f"[SPLIT] unused_count: {len(unused_names)}")
        print(f"[SPLIT] overlap    : {len(split_manifest['overlap_train_test'])}")
        print(f"[SPLIT] frame_mode : {frame_report.get('mode')}")
        print("-------------------------------------------------")

    elif eval and not sparse_split_active and not full_test_split_active:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    if not use_external_eval:
        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
            images_folder=os.path.join(path, reading_dir),
            depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
        all_cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

        if sparse_split_active:
            train_cam_infos, test_cam_infos = _build_sparse_train_test_split(
                all_cam_infos=all_cam_infos,
                scene_path=path,
                sparse_train_images=sparse_train_images,
                sparse_train_indices=sparse_train_indices,
                sparse_train_count=sparse_train_count,
            )
            if full_test_split_active:
                test_cam_infos = []
        else:
            train_cam_infos = [c for c in all_cam_infos if train_test_exp or not c.is_test]
            test_cam_infos = [c for c in all_cam_infos if c.is_test]

        if full_test_split_active:
            full_test_cam_extrinsics, full_test_cam_intrinsics, _ = _read_colmap_extrinsics_intrinsics(full_test_source_path)
            full_test_reading_dir = full_test_images if full_test_images else reading_dir
            full_test_cam_infos_unsorted = readColmapCameras(
                cam_extrinsics=full_test_cam_extrinsics,
                cam_intrinsics=full_test_cam_intrinsics,
                depths_params=None,
                images_folder=os.path.join(full_test_source_path, full_test_reading_dir),
                depths_folder="",
                test_cam_names_list=[],
            )
            full_test_cam_infos = sorted(full_test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)
            train_cam_infos = [c._replace(is_test=False) for c in train_cam_infos]
            test_cam_infos = _build_full_source_test_split(train_cam_infos, full_test_cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False,
                           split_manifest=split_manifest)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True,
                           split_manifest={})
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
