#!/usr/bin/env python3
"""
YOLO26 cross-hardware benchmark: Apple Silicon (Core ML / Core AI / MPS) vs NVIDIA (TensorRT / CUDA).

Same models, same image, same input size, same NMS-free head on every device.
Each platform uses the fastest path Ultralytics documents for it.

Usage (examples):
  python bench.py --label "M4 Air 16GB"
  python bench.py --label "M4 Pro 36GB" --power --soak-minutes 30
  python bench.py --label "Jetson Orin NX 16GB" --power --calib-data coco128.yaml
"""
import argparse
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION
from ultralytics.utils import ASSETS

IMGSZ = 640
PERSON = [0]


# ----------------------------------------------------------------------------- platform

def detect_platform():
    if sys.platform == "darwin":
        return "mac"
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    try:
        import torch
        if torch.cuda.is_available():
            return "nvidia"
    except Exception:
        pass
    return "cpu"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def hardware_info(plat):
    info = {"hostname": socket.gethostname(), "python": platform.python_version()}
    if plat == "mac":
        info["chip"] = sh("sysctl -n machdep.cpu.brand_string")
        info["memory_gb"] = round(int(sh("sysctl -n hw.memsize") or 0) / 1e9, 1)
        info["macos"] = platform.mac_ver()[0]
        info["model_id"] = sh("sysctl -n hw.model")
    else:
        try:
            import torch
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        except Exception:
            pass
        info["driver"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
        if plat == "jetson":
            info["jetson_model"] = sh("tr -d '\\0' < /proc/device-tree/model")
            info["l4t"] = sh("head -n 1 /etc/nv_tegra_release")
            info["nvpmodel"] = sh("nvpmodel -q 2>/dev/null | head -n 1")
        info["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    return info


def library_versions():
    v = {"ultralytics": ULTRALYTICS_VERSION}
    for name in ("torch", "coremltools", "tensorrt", "onnx"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            pass
    return v


def macos_major():
    try:
        return int(platform.mac_ver()[0].split(".")[0])
    except Exception:
        return 0


# ----------------------------------------------------------------------------- power sampling

class PowerSampler:
    """Background sampler. Each sample: {'t': time, 'w': total watts, ...extras}."""

    def __init__(self, plat):
        self.plat = plat
        self.samples = []
        self.proc = None
        self.thread = None
        self.source = None

    def start(self):
        if self.plat == "mac":
            # powermetrics needs root; cache sudo credentials first.
            print("[power] powermetrics needs sudo; you may be asked for your password.")
            subprocess.run(["sudo", "-v"])
            cmd = ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power,thermal", "-i", "500"]
            self.source = "powermetrics: CPU+GPU+ANE package power (not wall power)"
        elif self.plat == "jetson":
            cmd = ["tegrastats", "--interval", "500"]
            self.source = "tegrastats: VDD_IN board input if present, else sum of rails (approx)"
        elif self.plat == "nvidia":
            cmd = ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu", "--format=csv,noheader,nounits", "-lms", "500"]
            self.source = "nvidia-smi: GPU board power only (excludes CPU/system)"
        else:
            return
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except FileNotFoundError:
            print(f"[power] {cmd[0]} not found; power will not be recorded.")
            self.proc = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        cur = {}
        for line in self.proc.stdout:
            line = line.strip()
            now = time.time()
            if self.plat == "mac":
                for key, pat in (("cpu_mw", r"^CPU Power:\s*(\d+)\s*mW"),
                                 ("gpu_mw", r"^GPU Power:\s*(\d+)\s*mW"),
                                 ("ane_mw", r"^ANE Power:\s*(\d+)\s*mW")):
                    m = re.search(pat, line)
                    if m:
                        cur[key] = int(m.group(1))
                m = re.search(r"Current pressure level:\s*(\w+)", line)
                if m:
                    cur["thermal"] = m.group(1)
                m = re.search(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW", line)
                if m:
                    self.samples.append({"t": now, "w": int(m.group(1)) / 1000.0, **cur})
            elif self.plat == "jetson":
                rails = {k: int(v) for k, v, _ in re.findall(r"(\w+) (\d+)mW/(\d+)mW", line)}
                if not rails:
                    continue
                total = rails.get("VDD_IN", sum(rails.values()))
                temps = dict(re.findall(r"(\w+)@([\d.]+)C", line))
                gpu_t = next((float(v) for k, v in temps.items() if k.lower().startswith("gpu")), None)
                self.samples.append({"t": now, "w": total / 1000.0, "rails_mw": rails, "temp_c": gpu_t})
            elif self.plat == "nvidia":
                parts = [p.strip() for p in line.split(",")]
                try:
                    self.samples.append({"t": now, "w": float(parts[0]), "temp_c": float(parts[1])})
                except (ValueError, IndexError):
                    pass

    def window(self, t0, t1):
        s = [x for x in self.samples if t0 <= x["t"] <= t1]
        if not s:
            return {}
        out = {"power_w_avg": round(statistics.mean(x["w"] for x in s), 2), "power_samples": len(s)}
        for k in ("cpu_mw", "gpu_mw", "ane_mw", "temp_c"):
            vals = [x[k] for x in s if x.get(k) is not None]
            if vals:
                out[f"{k}_avg"] = round(statistics.mean(vals), 1)
        th = [x["thermal"] for x in s if x.get("thermal")]
        if th:
            out["thermal_states"] = sorted(set(th))
        return out

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


# ----------------------------------------------------------------------------- export

def export_model(pt, fmt, quantize, batch, out_dir, extra=None):
    """Export with the current `quantize` API, falling back to legacy half/int8 flags."""
    kwargs = dict(format=fmt, imgsz=IMGSZ, nms=False, batch=batch)
    if batch > 1 and fmt != "coreai":
        kwargs["dynamic"] = True  # CoreML requires dynamic for batch > 1; same setting on TensorRT for parity
    kwargs.update(extra or {})
    model = YOLO(pt)
    try:
        path = model.export(quantize=quantize, **kwargs)
    except Exception as e:
        if "quantize" not in str(e):
            raise
        legacy = {"half": True} if quantize == 16 else {"int8": True}
        path = YOLO(pt).export(**legacy, **kwargs)
    src = Path(path)
    stem = Path(pt).stem
    dst = out_dir / f"{stem}_{fmt}_q{quantize}_b{batch}{src.suffix}"
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


# ----------------------------------------------------------------------------- measurement

def time_predict(model, img, batch, device, warmup, runs):
    src = [img] * batch if batch > 1 else img
    kw = dict(imgsz=IMGSZ, classes=PERSON, verbose=False)
    if device is not None:
        kw["device"] = device
    for _ in range(warmup):
        res = model.predict(src, **kw)
    verified = len(res) == batch
    calls, pre, inf, post = [], [], [], []
    for _ in range(runs):
        t = time.perf_counter()
        res = model.predict(src, **kw)
        calls.append((time.perf_counter() - t) * 1000)
        sp = res[0].speed
        pre.append(sp.get("preprocess", 0))
        inf.append(sp.get("inference", 0))
        post.append(sp.get("postprocess", 0))
    calls.sort()
    mean = statistics.mean(calls)
    return {
        "call_ms": {
            "mean": round(mean, 3),
            "p50": round(calls[len(calls) // 2], 3),
            "p90": round(calls[int(len(calls) * 0.9) - 1], 3),
            "p99": round(calls[max(int(len(calls) * 0.99) - 1, 0)], 3),
            "std": round(statistics.pstdev(calls), 3),
        },
        "per_image_ms": round(mean / batch, 3),
        "throughput_ips": round(1000 * batch / mean, 1),
        "ultralytics_speed_ms": {
            "preprocess": round(statistics.mean(pre), 3),
            "inference": round(statistics.mean(inf), 3),
            "postprocess": round(statistics.mean(post), 3),
        },
        "batch_verified": verified,
    }


def validate(model, data, device):
    kw = dict(data=data, imgsz=IMGSZ, batch=1, plots=False, verbose=False)
    if device is not None:
        kw["device"] = device
    m = model.val(**kw)
    out = {"map50_95_all": round(float(m.box.map), 4)}
    try:
        out["map50_95_person"] = round(float(m.box.maps[0]), 4)
    except Exception:
        pass
    return out


def soak(model, img, device, minutes, sampler, window_s=30):
    kw = dict(imgsz=IMGSZ, classes=PERSON, verbose=False)
    if device is not None:
        kw["device"] = device
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    end = time.time() + minutes * 60
    windows, errors, lat = [], 0, []
    w_start, w_count = time.time(), 0
    while time.time() < end:
        t = time.perf_counter()
        try:
            model.predict(img, **kw)
            lat.append((time.perf_counter() - t) * 1000)
            w_count += 1
        except Exception:
            errors += 1
        if time.time() - w_start >= window_s:
            w_end = time.time()
            win = {"t_min": round((w_end - (end - minutes * 60)) / 60, 2),
                   "fps": round(w_count / (w_end - w_start), 1)}
            win.update(sampler.window(w_start, w_end) if sampler else {})
            windows.append(win)
            print(f"  [soak] {win['t_min']:>6} min  {win['fps']:>7} fps  {win.get('power_w_avg', '-')} W")
            w_start, w_count = w_end, 0
    lat.sort()
    first, last = (windows[0]["fps"], windows[-1]["fps"]) if windows else (None, None)
    return {
        "minutes": minutes,
        "errors": errors,
        "frames": len(lat),
        "latency_p50_ms": round(lat[len(lat) // 2], 3) if lat else None,
        "latency_p99_ms": round(lat[max(int(len(lat) * 0.99) - 1, 0)], 3) if lat else None,
        "fps_first_window": first,
        "fps_last_window": last,
        "fps_drop_pct": round(100 * (first - last) / first, 1) if first else None,
        "rss_growth_mb": round((proc.memory_info().rss - rss0) / 1e6, 1),
        "windows": windows,
    }


# ----------------------------------------------------------------------------- plan

def build_plan(plat, args):
    """Return list of (format_key, export_fmt or None, quantize, device, extra_export_kwargs)."""
    plan = []
    if plat == "mac":
        plan.append(("pytorch-mps-fp32", None, None, "mps", None))
        plan.append(("coreml-fp16", "coreml", 16, None, None))
        if not args.skip_int8:
            plan.append(("coreml-int8w", "coreml", 8, None, None))
        if macos_major() >= 27 and not args.skip_coreai:
            plan.append(("coreai-fp16", "coreai", 16, None, None))
    elif plat in ("nvidia", "jetson"):
        plan.append(("pytorch-cuda-fp32", None, None, 0, None))
        plan.append(("tensorrt-fp16", "engine", 16, 0, None))
        if not args.skip_int8:
            plan.append(("tensorrt-int8", "engine", 8, 0, {"data": args.calib_data}))
        if plat == "jetson" and args.dla:
            plan.append(("tensorrt-dla-fp16", "engine", 16, "dla:0", {"device": "dla:0"}))
    else:
        plan.append(("pytorch-cpu-fp32", None, None, "cpu", None))
    return plan


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help='Device label used in reports, e.g. "M4 Air 16GB"')
    ap.add_argument("--models", default="yolo26n,yolo26s,yolo26m")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--val-data", default="coco128.yaml", help="Dataset for accuracy (batch-1 exports only)")
    ap.add_argument("--calib-data", default="coco128.yaml", help="TensorRT INT8 calibration dataset")
    ap.add_argument("--skip-val", action="store_true")
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--skip-coreai", action="store_true")
    ap.add_argument("--dla", action="store_true", help="Jetson only: also test DLA (TensorRT <= 10.7)")
    ap.add_argument("--power", action="store_true", help="Sample power during runs")
    ap.add_argument("--soak-minutes", type=float, default=10, help="0 to skip the sustained run")
    ap.add_argument("--soak-model", default="yolo26s")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    plat = detect_platform()
    out_dir = Path(args.out)
    exp_dir = Path("exports")
    out_dir.mkdir(exist_ok=True)
    exp_dir.mkdir(exist_ok=True)

    print(f"Platform: {plat} | label: {args.label} | ultralytics {ULTRALYTICS_VERSION}")
    sampler = PowerSampler(plat) if args.power else None
    if sampler:
        sampler.start()
        time.sleep(2)

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    batches = [int(b) for b in args.batches.split(",")]
    report = {
        "label": args.label,
        "platform": plat,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware_info(plat),
        "versions": library_versions(),
        "settings": {k: v for k, v in vars(args).items()},
        "power_source": sampler.source if sampler else None,
        "results": [],
    }
    plan = build_plan(plat, args)
    soak_target = None

    try:
        for name in args.models.split(","):
            pt = f"{name}.pt"
            for key, fmt, q, device, extra in plan:
                for b in batches:
                    row = {"model": name, "format": key, "batch": b, "status": "ok"}
                    print(f"\n== {name} | {key} | batch {b}")
                    try:
                        if fmt is None:
                            path = pt
                        else:
                            t0 = time.time()
                            path = export_model(pt, fmt, q, b, exp_dir, extra)
                            row["export_s"] = round(time.time() - t0, 1)
                        row["model_path"] = str(path)
                        model = YOLO(str(path), task="detect")
                        t0 = time.time()
                        row.update(time_predict(model, img, b, device, args.warmup, args.runs))
                        t1 = time.time()
                        if sampler:
                            row.update(sampler.window(t0, t1))
                        if b == 1 and not args.skip_val:
                            row.update(validate(model, args.val_data, device))
                        print(f"   {row['per_image_ms']} ms/img  {row['throughput_ips']} img/s"
                              f"  {row.get('power_w_avg', '-')} W  ANE {row.get('ane_mw_avg', '-')} mW")
                        if b == 1 and name == args.soak_model and key.split("-")[0] in ("coreml", "tensorrt") \
                                and "fp16" in key and soak_target is None:
                            soak_target = (key, str(path), device)
                    except Exception as e:
                        row["status"] = "error"
                        row["error"] = f"{type(e).__name__}: {e}"[:500]
                        print(f"   ERROR {row['error']}")
                    report["results"].append(row)

        if args.soak_minutes > 0 and soak_target:
            key, path, device = soak_target
            print(f"\n== Soak: {args.soak_model} | {key} | batch 1 | {args.soak_minutes} min")
            report["soak"] = {"model": args.soak_model, "format": key,
                              **soak(YOLO(path, task="detect"), img, device, args.soak_minutes, sampler)}
    finally:
        if sampler:
            sampler.stop()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", args.label).strip("_")
        fname = out_dir / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.json"
        fname.write_text(json.dumps(report, indent=2))
        print(f"\nSaved {fname}")


if __name__ == "__main__":
    main()
