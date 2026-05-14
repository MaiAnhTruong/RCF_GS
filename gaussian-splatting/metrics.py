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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from argparse import ArgumentParser
from utils.metric_utils import MetricEvaluator, summarize_metrics

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    metric_evaluator = MetricEvaluator(lpips_net_type="vgg")
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                metric_values = {"SSIM": [], "PSNR": [], "LPIPS": []}

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    per_view_metrics = metric_evaluator.evaluate(renders[idx], gts[idx])
                    for metric_name, metric_value in per_view_metrics.items():
                        metric_values[metric_name].append(metric_value)

                mean_metrics = summarize_metrics(metric_values)

                print("  SSIM : {:>12.7f}".format(mean_metrics["SSIM"], ".5"))
                print("  PSNR : {:>12.7f}".format(mean_metrics["PSNR"], ".5"))
                print("  LPIPS: {:>12.7f}".format(mean_metrics["LPIPS"], ".5"))
                print("")

                full_dict[scene_dir][method].update(mean_metrics)
                per_view_dict[scene_dir][method].update({"SSIM": {name: value for value, name in zip(metric_values["SSIM"], image_names)},
                                                            "PSNR": {name: value for value, name in zip(metric_values["PSNR"], image_names)},
                                                            "LPIPS": {name: value for value, name in zip(metric_values["LPIPS"], image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
