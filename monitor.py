"""WhiRL server monitor using rich-based TUI.

The CPU usage is computed as `[CPU load average] / [number of CPUs]` and
presented as a colored bar like in htop (e.g. `[||         16.2%]`). The GPU
info column presents an array where each element correspond to a machine's GPU
 and the element's color indicates that GPU's utilization. The colors change
from `white -> green -> yellow -> red` corresponding to utilization thresholds
of `0%, 25%, 50%, 75%`.

Known problems:
- If your ssh requests ProxyCommand/ProxyJump through linux.cs.ox.ac.uk, you
  will most likely see a lot of ERROR's in the table. This is due to the host
  limiting the number of parallel ssh connections. A solution would be to
  detect this and then fetch all the machines through linux host within the
  same call. Current workaround is to use VPN or connect directly to Oxford
  network so that the proxy is not needed.
- On window resize, the table's cells may overflow until the next update
  happens. Could be fixed by creating a mechanism which immediately updates
  the table on resize.
- No handling for cases where the list size is greater than the window width.
  A fix solution would be to let the list be scrollable.
"""

import argparse
import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
import math
import re
import shlex
import time
from typing import Optional

import pandas as pd
import rich
from rich.console import Console
from rich.live import Live
from rich.table import Table


def color_for_usage_fraction(fraction: float) -> str:
    assert 0.0 <= fraction <= 1.0, fraction
    thresholds_colors = (
        (0.00, 'bright_white'),
        (0.25, 'green1'),
        (0.50, 'yellow'),
        (0.75, 'red'),
    )

    color = next((color for threshold, color in thresholds_colors[::-1]
                  if threshold <= fraction))
    return color


@dataclass
class GpuInfo:
    data: Optional[pd.DataFrame]

    def usage_str(self, width: int) -> str:
        if self.data is None:
            return "[white on red]ERROR[/white on red]"

        usage_str_parts = []
        for _, i, usage_percent in self.data[['index', 'utilization.gpu']].itertuples():
            usage_fraction = usage_percent / 100.0
            color = color_for_usage_fraction(usage_fraction)

            gpu_usage_str = " ".join([
                f"[{color}]{i}[/{color}]"
            ])
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



@dataclass
class CpuInfo:
    usage_counts: Optional[Counter]
    load_avg: Optional[float]
    num_cpus: Optional[int]

    def usage_str(self, width: int) -> str:
        if (self.usage_counts is None or
            self.load_avg is None or
            self.num_cpus is None):
            return "[white on red]ERROR[/white on red]"

        visual_usage_length = width
        usage_fraction = self.load_avg / self.num_cpus
        usage_str = f"{usage_fraction * 100:.1f}%"
        usage_space = visual_usage_length - len(usage_str)
        usage_bars = min(
            int(math.ceil(usage_space * usage_fraction)), usage_space)

        color = color_for_usage_fraction(usage_fraction)

        visual_usage_str = "".join((
            r"[",
            (f"[{color}]{'|' * usage_bars}[/{color}]"),
            (" " * (usage_space - usage_bars)),
            usage_str,
            "]"
        ))

        return visual_usage_str


async def run_command(command):
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)


    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
        returncode = process.returncode
    except asyncio.TimeoutError:
        stdout = b""
        stderr = b""
        returncode = None

    return {
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "returncode": returncode,
    }


async def fetch_gpu_info(hostname: str) -> pd.DataFrame:
    command_parts = (
        "ssh",
        hostname,
        "nvidia-smi",
        "--query-gpu=" + ','.join((
            "index",
            "gpu_name",
            "memory.total",
            "memory.used",
            "memory.free",
            "utilization.gpu",
            "utilization.memory",
         )),
        "--format=csv",
    )
    gpu_info = await run_command(command_parts)

    gpu_info = parse_gpu_info(gpu_info)
    return gpu_info


async def fetch_cpu_info(hostname: str) -> Optional[CpuInfo]:
    top_command_parts = (
        "ssh",
        hostname,
        "top",
        "-b",
        "-n1",
    )

    num_cpus_command_parts = (
        "ssh",
        hostname,
        "grep -c",
        shlex.quote("^processor"),
        shlex.quote("/proc/cpuinfo"),
    )

    top_result, num_cpus_result = await asyncio.gather(
        run_command(top_command_parts),
        run_command(num_cpus_command_parts),
    )

    if num_cpus_result['returncode'] != 0 or top_result['returncode'] != 0:
        cpu_info = CpuInfo(usage_counts=None, load_avg=None, num_cpus=None)

        return cpu_info

    num_cpus = int(num_cpus_result['stdout'].strip("\n"))
    top_lines = top_result['stdout'].splitlines()

    load_avg = float(top_lines[0].split()[-1])

    usage_counts: Counter[dict[str, int]] = Counter()
    for top_line in top_lines[7:]:
        top_line_split = top_line.split()
        user = top_line_split[1]
        cpu_percentage = float(top_line_split[8])
        usage_counts[user] += int(cpu_percentage / (100.0 * float(num_cpus)))

    cpu_info = CpuInfo(
        usage_counts=usage_counts, load_avg=load_avg, num_cpus=num_cpus)

    return cpu_info


async def fetch_gpu_infos(hostnames):
    gpu_infos = await asyncio.gather(*[
        fetch_gpu_info(hostname) for hostname in hostnames
    ])
    return gpu_infos


async def fetch_host_info(hostname):
    cpu_info, gpu_info = await asyncio.gather(
        fetch_cpu_info(hostname),
        fetch_gpu_info(hostname),
    )

    return cpu_info, gpu_info


async def fetch_host_infos(hostnames):
    host_infos = await asyncio.gather(*[
        fetch_host_info(hostname) for hostname in hostnames
    ])
    return host_infos


def parse_gpu_info(raw_gpu_info):
    if raw_gpu_info['returncode'] != 0:
        return None

    def convert_MiB_str_to_float(str_value):
        assert isinstance(str_value, str), (type(str_value), str_value)
        value, unit = (
            re.match("\s*(\d+)\s+\[?([a-zA-z]+)\]?\s*", str_value).groups())
        if unit != "MiB":
            raise NotImplementedError(
                "TODO(hartikainen): generalize this to work with other sizes.")

        return float(value)

    def convert_percent_str_to_float(str_value):
        value = re.match("\s*(\d+)\s+\[?%\]?\s*", str_value).groups()[0]
        return float(value)

    column_names = pd.read_csv(
        StringIO(raw_gpu_info['stdout']),
        nrows=0,
        skipinitialspace=True,
    ).columns

    MiB_converters = {
        column_name: convert_MiB_str_to_float
        for column_name in column_names
        if '[MiB]' in column_name
    }
    percent_converters = {
        column_name: convert_percent_str_to_float
        for column_name in column_names
        if '[%]' in column_name
    }

    converters = {**MiB_converters, **percent_converters}

    dataframe = pd.read_csv(
        StringIO(raw_gpu_info['stdout']),
        converters=converters,
        skipinitialspace=True,
    ).rename(columns=lambda x: re.sub('\s+\[.*\]$', '', x))

    gpu_memory_used_fraction = (
        dataframe['memory.used'] / dataframe['memory.total'])
    gpu_memory_used_percentage = gpu_memory_used_fraction * 100.0
    dataframe['memory_allocated'] = gpu_memory_used_percentage

    return dataframe


def parse_gpu_infos(raw_gpu_infos):
    gpu_infos = type(raw_gpu_infos)((
        parse_gpu_info(raw_gpu_info)
        for raw_gpu_info in raw_gpu_infos
    ))
    return gpu_infos


@dataclass
class MachineInfo:
    name: str
    cpu: Optional[CpuInfo]
    gpu: Optional[GpuInfo]


def generate_machine_infos(hostnames: Sequence[str]):
    host_infos = asyncio.run(fetch_host_infos(hostnames))

    machine_infos = [
        MachineInfo(
            name=hostname,
            cpu=cpu_info,
            gpu=GpuInfo(data=gpu_info),
        )
        for hostname, (cpu_info, gpu_info) in zip(hostnames, host_infos)
    ]

    return machine_infos


def create_process_table(hostnames: Sequence[str], width: int, height: int) -> Table:

    machine_infos = generate_machine_infos(hostnames)

    hostname_header = "hostname"
    table = Table(
        hostname_header, "CPU %", "GPU compute %", box=rich.box.SIMPLE)

    gpu_compute_width = 3 + 2 * max(machine_info.gpu.usage_str_len
                                 for machine_info in machine_infos)
    hostname_width = 3 + max(map(len, [*hostnames, hostname_header]))
    cpu_width = width - (gpu_compute_width + hostname_width) - 3

    for machine_info in machine_infos:
        table.add_row(
            str(machine_info.name),
            machine_info.cpu.usage_str(width=cpu_width),
            machine_info.gpu.usage_str(width=gpu_compute_width),
        )

    return table


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('hostnames', type=str, nargs='+')
    return parser


def main():
    config = get_parser().parse_args()
    hostnames = config.hostnames

    console = Console()

    table = create_process_table(
        hostnames=hostnames,
        width=console.size.width - 2,
        height=console.size.height)

    with Live(table, console=console, screen=True, auto_refresh=True) as live:
        while True:
            table = create_process_table(
                hostnames=hostnames,
                width=console.size.width - 2,
                height=console.size.height)
            live.update(table)
            time.sleep(1)


if __name__ == '__main__':
    main()
