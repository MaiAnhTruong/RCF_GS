import json
from pathlib import Path

import torch

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")


def _ensure_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    return image


class MetricEvaluator:
    def __init__(self, lpips_net_type: str = "vgg", lpips_version: str = "0.1"):
        self.lpips_net_type = lpips_net_type
        self.lpips_version = lpips_version
        self._lpips_models = {}

    def _get_lpips_model(self, device) -> LPIPS:
        device = torch.device(device)
        device_key = str(device)
        if device_key not in self._lpips_models:
            model = LPIPS(self.lpips_net_type, self.lpips_version).to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self._lpips_models[device_key] = model
        return self._lpips_models[device_key]

    def warmup(self, device) -> None:
        self._get_lpips_model(device)

    @torch.no_grad()
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = _ensure_batched_image(prediction).detach()
        target = _ensure_batched_image(target).detach()
        lpips_model = self._get_lpips_model(prediction.device)

        return {
            "PSNR": float(psnr(prediction, target).mean().item()),
            "SSIM": float(ssim(prediction, target).item()),
            "LPIPS": float(lpips_model(prediction, target).mean().item()),
        }


def summarize_metrics(metric_values):
    summary = {}
    for metric_name in METRIC_NAMES:
        values = metric_values.get(metric_name, [])
        if values:
            summary[metric_name] = float(sum(values) / len(values))
        else:
            summary[metric_name] = 0.0
    return summary


class MetricLogWriter:
    def __init__(self, output_dir, filename: str = "metrics_log.jsonl", reset: bool = False):
        self.path = Path(output_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()

    def append(self, iteration: int, split_name: str, metrics, num_views: int) -> None:
        record = {
            "iteration": int(iteration),
            "split": split_name,
            "num_views": int(num_views),
            "metrics": {metric_name: float(metrics[metric_name]) for metric_name in METRIC_NAMES},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.write("\n")
