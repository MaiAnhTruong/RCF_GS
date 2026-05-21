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
import json
import math
import torch
from random import randint
from datetime import datetime
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from lpipsPyTorch import lpips
    LPIPS_AVAILABLE = True
except Exception:
    lpips = None
    LPIPS_AVAILABLE = False


def _as_float(x):
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().mean().item())
    return float(x)


def _safe_metric_value(x):
    try:
        value = _as_float(x)
        if math.isfinite(value):
            return value
        return float("nan")
    except Exception:
        return float("nan")


def _cuda_elapsed_ms(start_event, end_event):
    try:
        return start_event.elapsed_time(end_event)
    except RuntimeError as exc:
        if "not ready" not in str(exc).lower():
            raise
        end_event.synchronize()
        return start_event.elapsed_time(end_event)


def _format_float(x, digits=8):
    if x is None:
        return "nan"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "nan"
        return f"{x:.{digits}f}"
    except Exception:
        return "nan"


def _append_line(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _maybe_write_header(path, header):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        _append_line(path, header)


class TrainMetricsFileLogger:
    def __init__(
        self,
        model_path,
        text_filename="train_metrics.txt",
        tsv_filename="train_metrics.tsv",
        per_view_filename="per_view_metrics.tsv",
    ):
        self.model_path = model_path
        self.text_path = os.path.join(model_path, text_filename)
        self.tsv_path = os.path.join(model_path, tsv_filename)
        self.per_view_path = os.path.join(model_path, per_view_filename)

        os.makedirs(model_path, exist_ok=True)

        _maybe_write_header(
            self.tsv_path,
            "\t".join([
                "iteration",
                "timestamp",
                "split",
                "n_images",
                "num_gaussians",
                "elapsed_ms",
                "train_l1",
                "train_ssim",
                "train_dssim_loss",
                "train_rgb_loss",
                "train_depth_l1_pure",
                "train_depth_weight",
                "train_depth_loss",
                "train_total_loss",
                "eval_l1",
                "eval_mse",
                "eval_rmse",
                "eval_psnr",
                "eval_ssim",
                "eval_dssim_loss",
                "eval_rgb_loss",
                "eval_lpips_vgg",
            ])
        )

        _maybe_write_header(
            self.per_view_path,
            "\t".join([
                "iteration",
                "timestamp",
                "split",
                "image_name",
                "l1",
                "mse",
                "rmse",
                "psnr",
                "ssim",
                "dssim_loss",
                "rgb_loss",
                "lpips_vgg",
            ])
        )

    def _write_summary_row(self, iteration, timestamp, split_name, summary, train_scalars, num_gaussians, elapsed_ms):
        row = [
            iteration,
            timestamp,
            split_name,
            summary.get("n_images", 0),
            num_gaussians,
            _format_float(elapsed_ms, 6),
            _format_float(train_scalars.get("train_l1"), 8),
            _format_float(train_scalars.get("train_ssim"), 8),
            _format_float(train_scalars.get("train_dssim_loss"), 8),
            _format_float(train_scalars.get("train_rgb_loss"), 8),
            _format_float(train_scalars.get("train_depth_l1_pure"), 8),
            _format_float(train_scalars.get("train_depth_weight"), 8),
            _format_float(train_scalars.get("train_depth_loss"), 8),
            _format_float(train_scalars.get("train_total_loss"), 8),
            _format_float(summary.get("l1"), 8),
            _format_float(summary.get("mse"), 8),
            _format_float(summary.get("rmse"), 8),
            _format_float(summary.get("psnr"), 8),
            _format_float(summary.get("ssim"), 8),
            _format_float(summary.get("dssim_loss"), 8),
            _format_float(summary.get("rgb_loss"), 8),
            _format_float(summary.get("lpips_vgg"), 8),
        ]
        _append_line(self.tsv_path, "\t".join(map(str, row)))

    def write_iteration(self, iteration, train_scalars, eval_summaries, per_view_rows=None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        num_gaussians = train_scalars.get("num_gaussians", float("nan"))
        elapsed_ms = train_scalars.get("elapsed_ms", float("nan"))

        lines = []
        lines.append("=" * 80)
        lines.append(f"ITER {iteration} | {timestamp}")
        lines.append(f"num_gaussians = {num_gaussians}")
        lines.append(f"elapsed_ms    = {_format_float(elapsed_ms, 4)}")
        lines.append("")
        lines.append("TRAIN_PATCH")
        for key in [
            "train_l1",
            "train_ssim",
            "train_dssim_loss",
            "train_rgb_loss",
            "train_depth_l1_pure",
            "train_depth_weight",
            "train_depth_loss",
            "train_total_loss",
        ]:
            lines.append(f"  {key:<20} = {_format_float(train_scalars.get(key), 8)}")

        for split_name, summary in eval_summaries.items():
            lines.append("")
            lines.append(f"EVAL {split_name}")
            lines.append(f"  n_images     = {summary.get('n_images', 0)}")
            lines.append(f"  l1           = {_format_float(summary.get('l1'), 8)}")
            lines.append(f"  mse          = {_format_float(summary.get('mse'), 8)}")
            lines.append(f"  rmse         = {_format_float(summary.get('rmse'), 8)}")
            lines.append(f"  psnr         = {_format_float(summary.get('psnr'), 8)}")
            lines.append(f"  ssim         = {_format_float(summary.get('ssim'), 8)}")
            lines.append(f"  dssim_loss   = {_format_float(summary.get('dssim_loss'), 8)}")
            lines.append(f"  rgb_loss     = {_format_float(summary.get('rgb_loss'), 8)}")
            lines.append(f"  lpips_vgg    = {_format_float(summary.get('lpips_vgg'), 8)}")

        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        if eval_summaries:
            for split_name, summary in eval_summaries.items():
                self._write_summary_row(
                    iteration,
                    timestamp,
                    split_name,
                    summary,
                    train_scalars,
                    num_gaussians,
                    elapsed_ms,
                )
        else:
            self._write_summary_row(
                iteration,
                timestamp,
                "train_patch",
                {},
                train_scalars,
                num_gaussians,
                elapsed_ms,
            )

        if per_view_rows:
            for r in per_view_rows:
                row = [
                    iteration,
                    timestamp,
                    r.get("split", ""),
                    r.get("image_name", ""),
                    _format_float(r.get("l1"), 8),
                    _format_float(r.get("mse"), 8),
                    _format_float(r.get("rmse"), 8),
                    _format_float(r.get("psnr"), 8),
                    _format_float(r.get("ssim"), 8),
                    _format_float(r.get("dssim_loss"), 8),
                    _format_float(r.get("rgb_loss"), 8),
                    _format_float(r.get("lpips_vgg"), 8),
                ]
                _append_line(self.per_view_path, "\t".join(map(str, row)))


@torch.no_grad()
def evaluate_camera_set_for_metrics(
    split_name,
    cameras,
    scene,
    renderFunc,
    renderArgs,
    lambda_dssim,
    train_test_exp,
    compute_lpips=True,
):
    if cameras is None or len(cameras) == 0:
        return None, []

    sums = {
        "l1": 0.0,
        "mse": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "dssim_loss": 0.0,
        "rgb_loss": 0.0,
        "lpips_vgg": 0.0,
    }

    valid_lpips_count = 0
    per_view_rows = []

    for viewpoint in cameras:
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        if train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]

        image_b = image.unsqueeze(0)
        gt_b = gt_image.unsqueeze(0)

        l1_v = torch.abs(image - gt_image).mean()
        mse_v = torch.mean((image - gt_image) ** 2)
        rmse_v = torch.sqrt(torch.clamp(mse_v, min=1e-12))
        psnr_v = psnr(image_b, gt_b).mean()
        ssim_v = ssim(image_b, gt_b)
        dssim_v = 1.0 - ssim_v
        rgb_loss_v = (1.0 - lambda_dssim) * l1_v + lambda_dssim * dssim_v

        lpips_v = torch.tensor(float("nan"), device="cuda")
        if compute_lpips and LPIPS_AVAILABLE and lpips is not None:
            try:
                lpips_v = lpips(image_b, gt_b, net_type="vgg").mean()
                valid_lpips_count += 1
            except Exception:
                lpips_v = torch.tensor(float("nan"), device="cuda")

        row = {
            "split": split_name,
            "image_name": getattr(viewpoint, "image_name", "unknown"),
            "l1": _safe_metric_value(l1_v),
            "mse": _safe_metric_value(mse_v),
            "rmse": _safe_metric_value(rmse_v),
            "psnr": _safe_metric_value(psnr_v),
            "ssim": _safe_metric_value(ssim_v),
            "dssim_loss": _safe_metric_value(dssim_v),
            "rgb_loss": _safe_metric_value(rgb_loss_v),
            "lpips_vgg": _safe_metric_value(lpips_v),
        }
        per_view_rows.append(row)

        sums["l1"] += row["l1"]
        sums["mse"] += row["mse"]
        sums["rmse"] += row["rmse"]
        sums["psnr"] += row["psnr"]
        sums["ssim"] += row["ssim"]
        sums["dssim_loss"] += row["dssim_loss"]
        sums["rgb_loss"] += row["rgb_loss"]

        if not math.isnan(row["lpips_vgg"]):
            sums["lpips_vgg"] += row["lpips_vgg"]

    n = len(cameras)
    summary = {
        "n_images": n,
        "l1": sums["l1"] / n,
        "mse": sums["mse"] / n,
        "rmse": sums["rmse"] / n,
        "psnr": sums["psnr"] / n,
        "ssim": sums["ssim"] / n,
        "dssim_loss": sums["dssim_loss"] / n,
        "rgb_loss": sums["rgb_loss"] / n,
        "lpips_vgg": sums["lpips_vgg"] / valid_lpips_count if valid_lpips_count > 0 else float("nan"),
    }

    return summary, per_view_rows

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    metrics_log_interval=100,
    metrics_eval_train_count=5,
    metrics_eval_per_view=False,
    metrics_disable_lpips=False,
):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    metrics_logger = TrainMetricsFileLogger(dataset.model_path)
    if not LPIPS_AVAILABLE and not metrics_disable_lpips:
        print("[WARN] LPIPS is not available. eval lpips_vgg will be written as nan.")
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        dssim_loss_value = 1.0 - ssim_value
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * dssim_loss_value
        loss = rgb_loss

        # Depth regularization
        depth_weight_now = depth_l1_weight(iteration)
        Ll1depth_pure = torch.tensor(0.0, device="cuda")
        Ll1depth_weighted = torch.tensor(0.0, device="cuda")

        if depth_weight_now > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth_weighted = depth_weight_now * Ll1depth_pure
            loss += Ll1depth_weighted

        Ll1depth = _safe_metric_value(Ll1depth_weighted)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            loss_item = loss.item()
            elapsed_ms = _cuda_elapsed_ms(iter_start, iter_end)
            train_scalars = {
                "train_l1": _safe_metric_value(Ll1),
                "train_ssim": _safe_metric_value(ssim_value),
                "train_dssim_loss": _safe_metric_value(dssim_loss_value),
                "train_rgb_loss": _safe_metric_value(rgb_loss),
                "train_depth_l1_pure": _safe_metric_value(Ll1depth_pure),
                "train_depth_weight": float(depth_weight_now),
                "train_depth_loss": _safe_metric_value(Ll1depth_weighted),
                "train_total_loss": _safe_metric_value(loss),
                "elapsed_ms": _safe_metric_value(elapsed_ms),
                "num_gaussians": int(gaussians.get_xyz.shape[0]),
            }

            # Progress bar
            ema_loss_for_log = 0.4 * loss_item + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                elapsed_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                train_scalars=train_scalars,
                metrics_logger=metrics_logger,
                lambda_dssim=opt.lambda_dssim,
                metrics_log_interval=metrics_log_interval,
                metrics_eval_train_count=metrics_eval_train_count,
                metrics_eval_per_view=metrics_eval_per_view,
                metrics_disable_lpips=metrics_disable_lpips,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp,
    train_scalars=None,
    metrics_logger=None,
    lambda_dssim=0.2,
    metrics_log_interval=100,
    metrics_eval_train_count=5,
    metrics_eval_per_view=False,
    metrics_disable_lpips=False,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if tb_writer and train_scalars is not None:
        tb_writer.add_scalar("train_loss_patches/ssim", train_scalars["train_ssim"], iteration)
        tb_writer.add_scalar("train_loss_patches/dssim_loss", train_scalars["train_dssim_loss"], iteration)
        tb_writer.add_scalar("train_loss_patches/rgb_loss", train_scalars["train_rgb_loss"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_l1_pure", train_scalars["train_depth_l1_pure"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_weight", train_scalars["train_depth_weight"], iteration)
        tb_writer.add_scalar("train_loss_patches/depth_loss", train_scalars["train_depth_loss"], iteration)
        tb_writer.add_scalar("scene/total_points", train_scalars["num_gaussians"], iteration)

    should_eval = iteration in testing_iterations
    should_train_log = metrics_log_interval > 0 and (iteration % metrics_log_interval == 0)

    if should_eval:
        torch.cuda.empty_cache()
        first_testing_iteration = min((test_iteration for test_iteration in testing_iterations if test_iteration > 0), default=None)

        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()

        if metrics_eval_train_count < 0:
            train_eval_cameras = train_cameras
            train_split_name = "train_full"
        elif metrics_eval_train_count == 0 or len(train_cameras) == 0:
            train_eval_cameras = []
            train_split_name = "train_sample"
        else:
            train_eval_cameras = [
                train_cameras[idx % len(train_cameras)]
                for idx in range(5, 5 * (metrics_eval_train_count + 1), 5)
            ]
            train_split_name = "train_sample"

        validation_configs = [
            {"name": "test", "cameras": test_cameras},
            {"name": train_split_name, "cameras": train_eval_cameras},
        ]

        eval_summaries = {}
        all_per_view_rows = []

        compute_lpips = (not metrics_disable_lpips) and LPIPS_AVAILABLE

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                summary, per_view_rows = evaluate_camera_set_for_metrics(
                    split_name=config["name"],
                    cameras=config["cameras"],
                    scene=scene,
                    renderFunc=renderFunc,
                    renderArgs=renderArgs,
                    lambda_dssim=lambda_dssim,
                    train_test_exp=train_test_exp,
                    compute_lpips=compute_lpips,
                )

                if summary is not None:
                    eval_summaries[config["name"]] = summary

                    if metrics_eval_per_view:
                        all_per_view_rows.extend(per_view_rows)

                    print(
                        "\n[ITER {}] Evaluating {}: "
                        "L1 {:.6f} MSE {:.6f} RMSE {:.6f} PSNR {:.6f} SSIM {:.6f} LPIPS {}".format(
                            iteration,
                            config["name"],
                            summary["l1"],
                            summary["mse"],
                            summary["rmse"],
                            summary["psnr"],
                            summary["ssim"],
                            _format_float(summary["lpips_vgg"], 6),
                        )
                    )

                    if tb_writer:
                        for idx, viewpoint in enumerate(config["cameras"][:5]):
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            if train_test_exp:
                                image = image[..., image.shape[-1] // 2:]
                                gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                            tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if first_testing_iteration is not None and iteration == first_testing_iteration:
                                tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                        tb_writer.add_scalar(config["name"] + "/l1", summary["l1"], iteration)
                        tb_writer.add_scalar(config["name"] + "/mse", summary["mse"], iteration)
                        tb_writer.add_scalar(config["name"] + "/rmse", summary["rmse"], iteration)
                        tb_writer.add_scalar(config["name"] + "/psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/ssim", summary["ssim"], iteration)
                        tb_writer.add_scalar(config["name"] + "/dssim_loss", summary["dssim_loss"], iteration)
                        tb_writer.add_scalar(config["name"] + "/rgb_loss", summary["rgb_loss"], iteration)
                        if not math.isnan(summary["lpips_vgg"]):
                            tb_writer.add_scalar(config["name"] + "/lpips_vgg", summary["lpips_vgg"], iteration)
                            tb_writer.add_scalar(config["name"] + "/metrics/lpips", summary["lpips_vgg"], iteration)

                        tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", summary["l1"], iteration)
                        tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/metrics/psnr", summary["psnr"], iteration)
                        tb_writer.add_scalar(config["name"] + "/metrics/ssim", summary["ssim"], iteration)

        if metrics_logger is not None and train_scalars is not None:
            metrics_logger.write_iteration(
                iteration=iteration,
                train_scalars=train_scalars,
                eval_summaries=eval_summaries,
                per_view_rows=all_per_view_rows if metrics_eval_per_view else None,
            )

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()

    elif should_train_log and metrics_logger is not None and train_scalars is not None:
        metrics_logger.write_iteration(
            iteration=iteration,
            train_scalars=train_scalars,
            eval_summaries={},
            per_view_rows=None,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--metrics_log_interval",
        type=int,
        default=100,
        help="Write train patch loss scalars to train_metrics.txt/tsv every N iterations. Set 0 to disable periodic train-only file logging."
    )
    parser.add_argument(
        "--metrics_eval_train_count",
        type=int,
        default=5,
        help="Number of train cameras to evaluate at test_iterations. Use -1 for all train cameras, 0 to disable train eval."
    )
    parser.add_argument(
        "--metrics_eval_per_view",
        action="store_true",
        default=False,
        help="Write per-view metrics to per_view_metrics.tsv at test_iterations."
    )
    parser.add_argument(
        "--metrics_disable_lpips",
        action="store_true",
        default=False,
        help="Disable LPIPS during training eval. Useful when LPIPS is unavailable or too slow."
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        metrics_log_interval=args.metrics_log_interval,
        metrics_eval_train_count=args.metrics_eval_train_count,
        metrics_eval_per_view=args.metrics_eval_per_view,
        metrics_disable_lpips=args.metrics_disable_lpips,
    )

    # All done
    print("\nTraining complete.")
