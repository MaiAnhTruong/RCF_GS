import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scene.dataset_readers import (
    _align_eval_cameras_to_train_frame,
    _check_same_frame_by_common_cameras,
    _filter_cam_infos_by_names,
    _read_colmap_extrinsics_intrinsics,
    _read_name_list_txt,
    _select_eval_test_names,
    readColmapCameras,
)


def _basename_set(names):
    return set(os.path.basename(n) for n in names)


def _read_cam_infos(source_path, images_dir, test_names=None):
    extrinsics, intrinsics, sparse_format = _read_colmap_extrinsics_intrinsics(source_path)
    cam_infos = readColmapCameras(
        cam_extrinsics=extrinsics,
        cam_intrinsics=intrinsics,
        depths_params=None,
        images_folder=os.path.join(source_path, images_dir),
        depths_folder="",
        test_cam_names_list=test_names or [],
    )
    return sorted(cam_infos, key=lambda c: c.image_name), extrinsics, sparse_format


def main():
    parser = argparse.ArgumentParser(description="Audit DPCR sparse-train / external-eval split.")
    parser.add_argument("--train_source", required=True)
    parser.add_argument("--eval_source", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--eval_images", default="")
    parser.add_argument("--split_mode", default="llffhold", choices=["llffhold", "test_txt", "manifest"])
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--train_view_list", default="")
    parser.add_argument("--test_view_list", default="")
    parser.add_argument("--frame_mode", default="strict", choices=["strict", "align_umeyama", "skip"])
    parser.add_argument("--alignment_min_common", type=int, default=4)
    parser.add_argument("--frame_check_tol", type=float, default=1e-3)
    args = parser.parse_args()

    train_source = os.path.abspath(args.train_source)
    eval_source = os.path.abspath(args.eval_source)
    eval_images = args.eval_images if args.eval_images else args.images

    train_all_cam_infos, train_extrinsics, train_sparse_format = _read_cam_infos(train_source, args.images)
    eval_all_cam_infos, eval_extrinsics, eval_sparse_format = _read_cam_infos(eval_source, eval_images)

    test_names, eval_all_names = _select_eval_test_names(
        eval_source_path=eval_source,
        eval_extrinsics=eval_extrinsics,
        split_mode=args.split_mode,
        llffhold=args.llffhold,
        test_view_list_path=args.test_view_list,
    )

    all_sparse_train_names = sorted(os.path.basename(train_extrinsics[k].name) for k in train_extrinsics)
    if args.train_view_list:
        train_names = _read_name_list_txt(args.train_view_list)
    else:
        train_names = all_sparse_train_names
    train_names = list(dict.fromkeys(os.path.basename(n) for n in train_names))

    missing_train = sorted(set(train_names) - set(all_sparse_train_names))
    if missing_train:
        raise ValueError(f"Train view list contains images not found in sparse train source: {missing_train[:20]}")

    train_cam_infos = sorted(
        _filter_cam_infos_by_names(train_all_cam_infos, include_names=train_names),
        key=lambda c: c.image_name,
    )
    test_cam_infos = sorted(
        _filter_cam_infos_by_names(eval_all_cam_infos, include_names=test_names),
        key=lambda c: c.image_name,
    )

    train_set = _basename_set(train_names)
    test_set = _basename_set(test_names)
    eval_all_set = _basename_set(eval_all_names)
    unused_names = sorted(eval_all_set - test_set - train_set)
    overlap = sorted(train_set & test_set)

    print("------------DPCR SPLIT AUDIT-------------")
    print(f"train_source : {train_source}")
    print(f"eval_source  : {eval_source}")
    print(f"train_sparse : {train_sparse_format}")
    print(f"eval_sparse  : {eval_sparse_format}")
    print(f"split_mode   : {args.split_mode}")
    print(f"llffhold     : {args.llffhold}")
    print(f"train_count  : {len(train_cam_infos)}")
    print(f"test_count   : {len(test_cam_infos)}")
    print(f"unused_count : {len(unused_names)}")
    print(f"overlap_count: {len(overlap)}")
    if overlap:
        print(f"overlap first 20: {overlap[:20]}")
    print(f"train first 20  : {sorted(train_set)[:20]}")
    print(f"test first 20   : {sorted(test_set)[:20]}")
    print("-----------------------------------------")

    if args.frame_mode == "strict":
        frame_report = _check_same_frame_by_common_cameras(
            train_cam_infos=train_cam_infos,
            eval_cam_infos=eval_all_cam_infos,
            min_common=args.alignment_min_common,
            tol=args.frame_check_tol,
        )
    elif args.frame_mode == "align_umeyama":
        _, frame_report = _align_eval_cameras_to_train_frame(
            train_cam_infos=train_cam_infos,
            eval_all_cam_infos=eval_all_cam_infos,
            test_cam_infos=test_cam_infos,
            min_common=args.alignment_min_common,
        )
    elif args.frame_mode == "skip":
        frame_report = {"mode": "skip", "warning": "coordinate frame check skipped"}
    else:
        raise ValueError(f"Unknown frame mode: {args.frame_mode}")

    print("coordinate frame report:")
    for key, value in frame_report.items():
        if key == "common_names":
            print(f"  {key}: {value[:20]}{' ...' if len(value) > 20 else ''}")
        else:
            print(f"  {key}: {value}")

    if overlap:
        raise SystemExit("Train/test overlap detected. Regenerate sparse train views from non-test images.")


if __name__ == "__main__":
    main()
