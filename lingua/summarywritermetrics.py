# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import logging
from collections import namedtuple
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lingua.distributed import get_is_master
from termcolor import colored 

logger = logging.getLogger()


@dataclass
class TensorboardArgs:
    log_dir: Optional[str] = None
    comment: Optional[str] = None  # Optional comment to append to the run name


@dataclass
class LoggingArgs:
    freq: int = 10  # Log every freq optimizer steps
    acc_freq: Optional[int] = None  # Log every acc_freq gradient accumulation steps

    tensorboard: Optional[TensorboardArgs] = None


class MetricLogger:
    def __init__(self, outdir: Path, args: Optional[Any] = None):
        self.outdir = outdir
        self.jsonl_writer = None
        self.args = args
        self.writer = None

    def open(self):
        if self.jsonl_writer is None:
            self.jsonl_writer = open(self.outdir, "a")
        if (
            self.args is not None
            and self.args.logging.tensorboard is not None
            and get_is_master()
        ):
            tb_args = self.args.logging.tensorboard
            log_dir = tb_args.log_dir if tb_args.log_dir else str(self.outdir.parent / 'runs')
            if tb_args.comment:
                log_dir = f"{log_dir}/{tb_args.comment}"
            self.writer = SummaryWriter(log_dir=log_dir)
            # Log config as text
            if self.args:
                config_dict = asdict(self.args)
                self.writer.add_text('config', json.dumps(config_dict, indent=2))

    def log(self, metrics: Dict[str, Any]):
        if self.writer:
            step = metrics.get("global_step", 0)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)
                elif isinstance(v, str):
                    self.writer.add_text(k, v, step)
                # Add more types as needed (e.g., histograms, images)

        metrics.update({"created_at": datetime.now(timezone.utc).isoformat()})
        print(json.dumps(metrics), file=self.jsonl_writer, flush=True)

    def close(self):
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
        "power_draw",
    ],
)


class GPUMemoryMonitor:
    """
    Class to monitor GPU memory usage
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        # reset stats, clear cache
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]
        power_draw = torch.cuda.power_draw()

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
            power_draw,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def __str__(self):
        mem_stats = self.get_peak_stats()
        display_str = f"{self.device_name} ({self.device_index}): {self.device_capacity_gib} GiB capacity, "
        display_str += (
            f"{mem_stats.max_reserved_gib} GiB peak, {mem_stats.max_reserved_pct}% peak"
        )
        return f"{display_str}"


def upload_train_to_tensorboard(
    ckpt_dir, log_dir=None, train=True, eval=True
):
    from omegaconf import OmegaConf
    import json
    from pathlib import Path
    from torch.utils.tensorboard import SummaryWriter

    cfg_path = Path(ckpt_dir) / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg_omega = OmegaConf.load(cfg_path)
        cfg = OmegaConf.to_container(cfg_omega)

    if log_dir is None:
        log_dir = str(Path(ckpt_dir) / 'tensorboard')

    writer = SummaryWriter(log_dir=log_dir)

    # Log config as text
    writer.add_text('config', json.dumps(cfg, indent=2))

    if train:
        metrics_path = Path(ckpt_dir) / "metrics.jsonl"
        if metrics_path.exists():
            with open(metrics_path) as f:
                for l in f:
                    m = json.loads(l)
                    step = m.get("global_step", 0)
                    for k, v in m.items():
                        if k not in ["global_step", "created_at"] and isinstance(v, (int, float)):
                            writer.add_scalar(k, v, step)

    if eval:
        eval_path = Path(ckpt_dir) / "metrics.eval.jsonl"
        if eval_path.exists():
            with open(eval_path) as f:
                for l in f:
                    m = json.loads(l)
                    step = m.get("global_step", 0)
                    for name, value in m.items():
                        if "/" in name and isinstance(value, (int, float)):
                            new_name = f"evals/{name.replace('/','.')}"
                            writer.add_scalar(new_name, value, step)

    writer.close()


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values()) 
