import asyncio


async def run_command(command):
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

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


def color_for_usage_fraction(fraction: float) -> str:
    assert 0.0 <= fraction, fraction
    thresholds_colors = (
        (0.00, "bright_white"),
        (0.25, "green1"),
        (0.50, "bright_yellow"),
        (0.75, "yellow"),
        (float("inf"), "red"),
    )

    color = next(
        (color for threshold, color in thresholds_colors if fraction <= threshold)
    )

    return color
