"""WhiRL server monitor using rich-based TUI.

The CPU usage is computed as `[CPU load average] / [number of CPUs]` and
presented as a colored bar like in htop (e.g. `[||         16.2%]`). The GPU
info column presents an array where each element correspond to a machine's GPU
 and the element's color indicates that GPU's utilization. The colors change
from `white -> green -> yellow -> orange -> red` corresponding to utilization
thresholds of `<=0%, <=25%, <=50%, <=75%, <=100%`.


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
from collections.abc import Sequence
from dataclasses import dataclass
import fileinput
import logging
import time
from typing import Optional

import rich
from rich.console import Console
from rich.live import Live
from rich.table import Table

from gpu import GpuFetcher, GpuInfo
from cpu import CpuFetcher, CpuInfo


def setup_logging(level):
    logging.basicConfig(filename="monitor.log", level=level)


async def fetch_host_info(hostname, fetchers):
    infos = await asyncio.gather(*[fetcher.fetch(hostname) for fetcher in fetchers])

    return infos


async def fetch_host_infos(hostnames, threads: int = -1):
    # It is important that we create the semaphore here rather than earlier
    # because otherwise the Semaphore's _loop variable is out of sync.
    # See https://stackoverflow.com/a/55918049 for more details.
    sem = asyncio.Semaphore(threads) if threads > 0 else None
    fetchers = [CpuFetcher(sem=sem), GpuFetcher(sem=sem)]
    host_infos = await asyncio.gather(
        *[fetch_host_info(hostname, fetchers) for hostname in hostnames]
    )
    return host_infos


@dataclass
class MachineInfo:
    name: str
    cpu: Optional[CpuInfo]
    gpu: Optional[GpuInfo]


def generate_machine_infos(hostnames: Sequence[str], threads: int = -1):
    host_infos = asyncio.run(fetch_host_infos(hostnames, threads=threads))

    machine_infos = [
        MachineInfo(
            name=hostname,
            cpu=cpu_info,
            gpu=GpuInfo(data=gpu_info),
        )
        for hostname, (cpu_info, gpu_info) in zip(hostnames, host_infos)
    ]

    return machine_infos


def create_process_table(
    hostnames: Sequence[str], width: int, height: int, threads: int = -1
) -> Table:

    machine_infos = generate_machine_infos(hostnames, threads=threads)

    hostname_header = "hostname"
    table = Table(hostname_header, "CPU %", "GPU compute %", box=rich.box.SIMPLE)

    gpu_compute_width = 3 + 2 * max(
        machine_info.gpu.usage_str_len for machine_info in machine_infos
    )
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
    parser.add_argument("hostnames", type=str, nargs="+")
    parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=-1,
    )
    parser.add_argument("--one-off", action="store_true", default=False)
    parser.add_argument("--interval", type=int, default=1)
    return parser


def main():
    config = get_parser().parse_args()
    setup_logging(level=logging.INFO)
    hostnames = config.hostnames
    console = Console()

    table = create_process_table(
        hostnames=hostnames,
        width=console.size.width - 2,
        height=console.size.height,
        threads=config.threads,
    )
    if config.one_off:
        console.print(table)
        return

    with Live(table, console=console, screen=True, auto_refresh=True) as live:
        while True:
            table = create_process_table(
                hostnames=hostnames,
                width=console.size.width - 2,
                height=console.size.height,
                threads=config.threads,
            )
            live.update(table)
            time.sleep(config.interval)


if __name__ == "__main__":
    main()
