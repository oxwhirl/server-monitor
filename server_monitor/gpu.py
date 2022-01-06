import pandas as pd
import re
import asyncio
import logging
from dataclasses import dataclass
from io import StringIO
from .fetcher_base import StatFetcher
from .utils import run_command, color_for_usage_fraction
from typing import Optional

LOG = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    data: Optional[pd.DataFrame]

    def usage_str(self, width: int) -> str:
        if self.data is None:
            return "[bright_white on red]ERROR[/bright_white on red]"

        usage_str_parts = []
        for _, i, usage_percent in self.data[["index", "utilization.gpu"]].itertuples():
            usage_fraction = usage_percent / 100.0
            color = color_for_usage_fraction(usage_fraction)

            gpu_usage_str = " ".join([f"[{color}]{i}[/{color}]"])
            usage_str_parts.append(gpu_usage_str)

        usage_str = " ".join(usage_str_parts)
        return usage_str

    @property
    def usage_str_len(self):
        if self.data is None:
            usage_str_size = len("ERROR")
        else:
            usage_str_size = self.data.shape[0]

        return usage_str_size


class GpuFetcher(StatFetcher):
    def __init__(self, sem=None):
        super(GpuFetcher, self).__init__(sem=sem)

    async def fetch_data(self, hostname: str) -> pd.DataFrame:
        command_parts = (
            "ssh",
            "-oBatchMode=yes",
            hostname,
            "nvidia-smi",
            "--query-gpu="
            + ",".join(
                (
                    "index",
                    "gpu_name",
                    "memory.total",
                    "memory.used",
                    "memory.free",
                    "utilization.gpu",
                    "utilization.memory",
                )
            ),
            "--format=csv",
        )
        gpu_info = await run_command(command_parts)

        gpu_info = parse_gpu_info(gpu_info)
        return gpu_info


def parse_gpu_info(raw_gpu_info):
    if raw_gpu_info["returncode"] != 0:
        LOG.error(f"Error fetching gpu info, return code {raw_gpu_info['returncode']}")
        LOG.error(f"Stdout: {raw_gpu_info['stdout']}")
        LOG.error(f"Stderr: {raw_gpu_info['stderr']}")
        return None

    def convert_MiB_str_to_float(str_value):
        assert isinstance(str_value, str), (type(str_value), str_value)
        value, unit = re.match("\s*(\d+)\s+\[?([a-zA-z]+)\]?\s*", str_value).groups()
        if unit != "MiB":
            raise NotImplementedError(
                "TODO(hartikainen): generalize this to work with other sizes."
            )

        return float(value)

    def convert_percent_str_to_float(str_value):
        value = re.match("\s*(\d+)\s+\[?%\]?\s*", str_value).groups()[0]
        return float(value)

    column_names = pd.read_csv(
        StringIO(raw_gpu_info["stdout"]),
        nrows=0,
        skipinitialspace=True,
    ).columns

    MiB_converters = {
        column_name: convert_MiB_str_to_float
        for column_name in column_names
        if "[MiB]" in column_name
    }
    percent_converters = {
        column_name: convert_percent_str_to_float
        for column_name in column_names
        if "[%]" in column_name
    }

    converters = {**MiB_converters, **percent_converters}

    dataframe = pd.read_csv(
        StringIO(raw_gpu_info["stdout"]),
        converters=converters,
        skipinitialspace=True,
    ).rename(columns=lambda x: re.sub("\s+\[.*\]$", "", x))

    gpu_memory_used_fraction = dataframe["memory.used"] / dataframe["memory.total"]
    gpu_memory_used_percentage = gpu_memory_used_fraction * 100.0
    dataframe["memory_allocated"] = gpu_memory_used_percentage

    return dataframe


def parse_gpu_infos(raw_gpu_infos):
    gpu_infos = type(raw_gpu_infos)(
        (parse_gpu_info(raw_gpu_info) for raw_gpu_info in raw_gpu_infos)
    )
    return gpu_infos
