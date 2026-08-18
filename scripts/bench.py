

"""Here we provide shared helpers for release timing scripts."""
from __future__ import annotations

import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy


def _run(command):
    """Return command output, or ``None`` when the query is unavailable."""
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _linux_core_count(text):
    """Count physical cores represented in Linux ``/proc/cpuinfo``."""
    pairs = set()
    for block in text.strip().split("\n\n"):
        physical = re.search(r"^physical id\s*:\s*(\d+)$", block, re.MULTILINE)
        core = re.search(r"^core id\s*:\s*(\d+)$", block, re.MULTILINE)
        if physical and core:
            pairs.add((physical.group(1), core.group(1)))
    return len(pairs) or None


def system_info():
    """Collect hardware and numerical-library metadata for a benchmark."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "logical_cores": os.cpu_count(),
    }

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(errors="ignore")
        model = re.findall(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        clocks = re.findall(r"^cpu MHz\s*:\s*([0-9.]+)$", text, re.MULTILINE)
        if model:
            info["cpu_model"] = model[0]
        physical = _linux_core_count(text)
        if physical is not None:
            info["physical_cores"] = physical
        if clocks:
            values = [float(value) for value in clocks]
            info["observed_mhz"] = [min(values), max(values)]

        mem = Path("/proc/meminfo")
        if mem.exists():
            match = re.search(
                r"^MemTotal:\s+(\d+)\s+kB", mem.read_text(), re.MULTILINE
            )
            if match:
                info["ram_bytes"] = int(match.group(1)) * 1024

        caches = []
        for path in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
            try:
                caches.append(
                    {
                        "level": (path / "level").read_text().strip(),
                        "type": (path / "type").read_text().strip(),
                        "size": (path / "size").read_text().strip(),
                        "shared": (path / "shared_cpu_list").read_text().strip(),
                    }
                )
            except OSError:
                pass
        if caches:
            info["caches"] = caches

    elif platform.system() == "Darwin":
        keys = {
            "cpu_model": "machdep.cpu.brand_string",
            "physical_cores": "hw.physicalcpu",
            "logical_cores": "hw.logicalcpu",
            "ram_bytes": "hw.memsize",
            "l1_data_bytes": "hw.l1dcachesize",
            "l2_bytes": "hw.l2cachesize",
            "l3_bytes": "hw.l3cachesize",
        }
        for name, key in keys.items():
            value = _run(["sysctl", "-n", key])
            if value is not None:
                try:
                    info[name] = int(value)
                except ValueError:
                    info[name] = value

    try:
        info["blas"] = np.__config__.CONFIG
    except AttributeError:
        info["blas"] = str(np.__config__.show())
    return info


def timed(function, repeats=7, warmups=2):
    """Measure repeated wall-clock runtimes after warm-up calls."""
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) * 1e-9)
    median = statistics.median(samples)
    return {
        "median_s": median,
        "mad_s": statistics.median(abs(value - median) for value in samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "repeats": repeats,
        "warmups": warmups,
    }
