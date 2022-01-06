import asyncio
from asyncio.runners import run
from typing import Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import logging
import math
import shlex
from .fetcher_base import StatFetcher
from .utils import color_for_usage_fraction, run_command

LOG = logging.getLogger(__name__)


@dataclass
class CpuInfo:
    usage_counts: Optional[Counter]
    load_avg: Optional[float]
    num_cpus: Optional[int]

    def usage_str(self, width: int) -> str:
        if self.usage_counts is None or self.load_avg is None or self.num_cpus is None:
            return "[bright_white on red]ERROR[/bright_white on red]"

        visual_usage_length = width
        usage_fraction = self.load_avg / self.num_cpus
        usage_str = f"{usage_fraction * 100:.1f}%"
        usage_space = visual_usage_length - len(usage_str)
        usage_bars = min(int(math.ceil(usage_space * usage_fraction)), usage_space)

        color = color_for_usage_fraction(usage_fraction)

        visual_usage_str = "".join(
            (
                r"[",
                (f"[{color}]{'|' * usage_bars}[/{color}]"),
                (" " * (usage_space - usage_bars)),
                usage_str,
                "]",
            )
        )

        return visual_usage_str


@dataclass
class RamInfo:
    usage_breakdown: Optional[Counter]
    total_memory: Optional[float]
    memory_usage: Optional[float]

    def usage_str(self, width: int) -> str:
        if (self.usage_breakdown is None
            or self.total_memory is None
            or self.memory_usage is None):
            return "[bright_white on red]ERROR[/bright_white on red]"

        visual_usage_length = width
        usage_fraction = self.memory_usage / self.total_memory
        usage_str = f"{usage_fraction * 100:.1f}%"
        usage_space = visual_usage_length - len(usage_str)
        usage_bars = min(int(math.ceil(usage_space * usage_fraction)), usage_space)

        color = color_for_usage_fraction(usage_fraction)

        visual_usage_str = "".join(
            (
                r"[",
                (f"[{color}]{'|' * usage_bars}[/{color}]"),
                (" " * (usage_space - usage_bars)),
                usage_str,
                "]",
            )
        )

        return visual_usage_str


class CpuAndRamFetcher(StatFetcher):
    def __init__(self, sem=None):
        super(CpuAndRamFetcher, self).__init__(sem=sem)

    async def fetch_data(self, hostname: str) -> Tuple[
            Optional[CpuInfo], Optional[RamInfo]]:
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

        if self.sem is None:
            top_result, num_cpus_result = await asyncio.gather(
                run_command(top_command_parts),
                run_command(num_cpus_command_parts),
            )
        else:
            # if the semaphore is not None, we have to avoid separate parallel
            # ssh connections because otherwise the parallelism is not as set
            # by the semaphore.
            top_result = await run_command(top_command_parts)
            num_cpus_result = await run_command(num_cpus_command_parts)

        if num_cpus_result["returncode"] != 0 or top_result["returncode"] != 0:

            cpu_info = CpuInfo(usage_counts=None, load_avg=None, num_cpus=None)
            LOG.error(
                f"num_cpus_result or top_result has a non-zero return code. \n"
                f"Num_cpus_result: {num_cpus_result['returncode']}\n"
                f"top_result: {top_result['returncode']}"
            )
            if num_cpus_result["returncode"] != 0:
                LOG.error(f"Num CPUs Stdout: {num_cpus_result['stdout']}")
                LOG.error(f"Num CPUs Stderr: {num_cpus_result['stderr']}")
            if top_result["returncode"] != 0:
                LOG.error(f"Top stdout: {top_result['stdout']}")
                LOG.error(f"Top stderr: {top_result['stderr']}")
            return cpu_info

        num_cpus = int(num_cpus_result["stdout"].strip("\n"))

        top_lines = top_result["stdout"].splitlines()

        load_avg = float(top_lines[0].split()[-1])

        memory_line_parts = top_lines[3].split()
        total_memory = float(memory_line_parts[3])
        memory_usage = float(memory_line_parts[-4])

        cpu_usage_counts: Counter[dict[str, int]] = Counter()
        ram_usage_breakdown: Counter[dict[str, float]] = Counter()
        for top_line in top_lines[7:]:
            top_line_split = top_line.split()
            user = top_line_split[1]
            cpu_percentage = float(top_line_split[8])
            ram_percentage = float(top_line_split[9])
            cpu_usage_counts[user] += int(cpu_percentage / (100.0 * float(num_cpus)))
            ram_usage_breakdown[user] += ram_percentage

        cpu_info = CpuInfo(
            usage_counts=cpu_usage_counts,
            load_avg=load_avg,
            num_cpus=num_cpus,
        )

        ram_info = RamInfo(
            usage_breakdown=ram_usage_breakdown,
            memory_usage=memory_usage,
            total_memory=total_memory)

        return cpu_info, ram_info