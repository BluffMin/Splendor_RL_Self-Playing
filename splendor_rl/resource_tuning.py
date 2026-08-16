from __future__ import annotations

import ctypes
import subprocess
import threading
from ctypes import wintypes


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD),
        ("memory_load", wintypes.DWORD),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _filetime_value(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def _system_times():
    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    )
    return tuple(_filetime_value(value) for value in (idle, kernel, user))


def _memory():
    value = _MemoryStatus()
    value.length = ctypes.sizeof(value)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return {
        "ram_percent": float(value.memory_load),
        "free_ram_gb": value.available_physical / 1024**3,
    }


def _gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        utilization, memory_mib, temperature = result.stdout.splitlines()[0].split(",")
        return {
            "gpu_percent": float(utilization),
            "vram_gb": float(memory_mib) / 1024,
            "gpu_temperature_c": float(temperature),
        }
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {}


class ResourceSampler:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        previous = _system_times()
        while not self._stop.wait(self.interval):
            current = _system_times()
            idle = current[0] - previous[0]
            total = (current[1] - previous[1]) + (current[2] - previous[2])
            cpu = 100.0 * (1.0 - idle / total) if total else 0.0
            previous = current
            self.samples.append({"cpu_percent": cpu, **_memory(), **_gpu()})

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 4)
        return self.summary()

    def summary(self):
        if not self.samples:
            return {}
        keys = set().union(*(sample.keys() for sample in self.samples))
        result = {"resource_samples": len(self.samples)}
        for key in keys:
            values = [sample[key] for sample in self.samples if key in sample]
            result[f"{key}_mean"] = sum(values) / len(values)
            result[f"{key}_peak"] = max(values)
            result[f"{key}_minimum"] = min(values)
        return result


def select_recommendation(rows, preset="balanced"):
    stable = [
        row
        for row in rows
        if row.get("workers", 0) > 0
        and row.get("illegal_actions") == 0
        and row.get("invariant_violations") == 0
    ]
    if not stable:
        raise ValueError("no stable multiprocess benchmark result")
    fastest = max(stable, key=lambda row: row["learning_transitions_per_second"])
    if preset == "maximum":
        return fastest
    safe = [
        row
        for row in stable
        if row.get("cpu_percent_peak", 0) <= 80
        and row.get("ram_percent_peak", 0) <= 80
        and row.get("vram_gb_peak", 0) <= 6
        and row.get("gpu_temperature_c_peak", 0) <= 75
        and row.get("free_ram_gb_minimum", 20) >= 20
    ]
    candidates = safe or stable
    best_speed = max(row["learning_transitions_per_second"] for row in candidates)
    near_best = [
        row
        for row in candidates
        if row["learning_transitions_per_second"] >= best_speed * 0.9
    ]
    return min(
        near_best,
        key=lambda row: (
            row.get("cpu_percent_mean", 100),
            row.get("ram_percent_peak", 100),
            row["workers"] * row["envs_per_worker"],
        ),
    )
