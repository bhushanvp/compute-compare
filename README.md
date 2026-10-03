# YOLO26 hardware benchmark: Apple Silicon vs NVIDIA

Same models (YOLO26n, s, m), same image, 640 input, NMS-free head, person class only.
Each device uses the fastest path Ultralytics documents for it.

| Platform | Formats tested |
|---|---|
| Mac | PyTorch MPS (reference), Core ML FP16, Core ML INT8 weights, Core AI FP16 (macOS 27 only) |
| RTX PC | PyTorch CUDA (reference), TensorRT FP16, TensorRT INT8 |
| Jetson | Same as RTX, plus optional DLA (`--dla`, Orin NX/AGX only) |

Every format runs at batch 1 (latency) and batch 8 (multi-stream throughput).
Accuracy is measured on batch-1 exports. One sustained run checks throttling and stability.

## 1. Setup (identical Ultralytics version everywhere)

Pick one Ultralytics version and install exactly that on every machine. The script records it.

**Macs**
```
python3 -m venv venv && source venv/bin/activate
pip install "ultralytics[export]==<same version>"
```

**RTX PC (Linux or Windows)**
```
pip install "ultralytics[export]==<same version>"   # needs CUDA PyTorch + TensorRT
```

**Jetson**: follow the Ultralytics NVIDIA Jetson guide for JetPack-matched PyTorch and TensorRT,
then install the same Ultralytics version. Use the TensorRT 10.x runtime from your JetPack.

## 2. Before each run

- Plug laptops into power. On the MacBook Pros, set Energy Mode to High Power.
- Close other apps, browsers and video calls.
- Jetson: `sudo nvpmodel -m 0 && sudo jetson_clocks` (max performance mode).
- Same room temperature if possible; the M4 Air is fanless and will throttle.

## 3. Run on each machine

```
python bench.py --label "M4 Air 16GB" --power
python bench.py --label "M3 Pro 18GB" --power
python bench.py --label "M4 Pro 36GB" --power --soak-minutes 30
python bench.py --label "Jetson Orin NX 16GB" --power
python bench.py --label "RTX 4060 PC" --power
```

Use your exact device names. Labels must match `prices.csv`.

Useful flags: `--runs 200`, `--soak-minutes 30`, `--skip-int8`, `--models yolo26n`, `--dla` (Jetson).
On Macs, `--power` asks for your sudo password once (powermetrics needs root).

Expect a long first run: every model is exported per precision and batch size,
and TensorRT INT8 calibration on a Jetson can take several minutes per model.

## 4. Compare

Copy every `results/*.json` into one `results/` folder, fill in `prices.csv`, then:
```
python compare.py
```
Outputs: `compare.md` (tables), `compare.csv`, `compare_throughput.png`.

## Reading the results honestly

- **FP16 is the fair cross-platform comparison.** Core ML INT8 quantizes weights only
  (FP16 activations); TensorRT INT8 quantizes activations too. Compare INT8 rows with their accuracy.
- **Power numbers aren't measured the same way.** Mac = CPU+GPU+ANE package power,
  RTX = GPU board only, Jetson = board input. For a true comparison, use a wall plug power meter.
- **Check `ane_mw` on Macs.** Near-zero ANE power means Core ML didn't run on the Neural Engine.
- **Check `batch_ok`.** If False, the format didn't really process all 8 images per call.
- **Price per device**: for Macs, use the price of the desktop you'd actually deploy (Mac mini/Studio),
  not the laptop. For NVIDIA, include the full system (PC or Jetson carrier board and PSU).
- **This measures model inference only**, not RTSP decode or a full pipeline.
- `coco128` is small; use a larger COCO val subset via `--val-data` for tighter accuracy numbers,
  and `--calib-data` with 500+ representative images for production-grade TensorRT INT8.
