#!/usr/bin/env python3
"""
Merge benchmark JSONs from every device into one comparison.

  python compare.py                      # reads results/*.json and prices.csv
  python compare.py --results results --prices prices.csv

Writes compare.csv, compare.md and compare_throughput.png.
"""
import argparse
import csv
import json
from pathlib import Path


def load_prices(path):
    prices = {}
    if Path(path).exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    prices[r["label"].strip()] = float(r["price_inr"])
                except (ValueError, KeyError, TypeError):
                    pass
    return prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prices", default="prices.csv")
    args = ap.parse_args()

    prices = load_prices(args.prices)
    rows, soaks = [], []
    for f in sorted(Path(args.results).glob("*.json")):
        rep = json.loads(f.read_text())
        label = rep["label"]
        price = prices.get(label)
        for r in rep["results"]:
            if r.get("status") != "ok":
                rows.append({"device": label, "model": r["model"], "format": r["format"], "batch": r["batch"],
                             "status": "error", "error": r.get("error", "")[:80]})
                continue
            ips = r["throughput_ips"]
            w = r.get("power_w_avg")
            rows.append({
                "device": label,
                "model": r["model"],
                "format": r["format"],
                "batch": r["batch"],
                "status": "ok",
                "ms_per_img": r["per_image_ms"],
                "p99_call_ms": r["call_ms"]["p99"],
                "img_per_s": ips,
                "watts": w,
                "img_per_s_per_w": round(ips / w, 2) if w else None,
                "ane_mw": r.get("ane_mw_avg"),
                "map_person": r.get("map50_95_person"),
                "map_all": r.get("map50_95_all"),
                "price_inr": price,
                "inr_per_img_s": round(price / ips, 1) if price and ips else None,
                "batch_ok": r.get("batch_verified"),
            })
        if rep.get("soak"):
            s = rep["soak"]
            soaks.append({"device": label, "model": s["model"], "format": s["format"], "minutes": s["minutes"],
                          "fps_first": s["fps_first_window"], "fps_last": s["fps_last_window"],
                          "drop_pct": s["fps_drop_pct"], "p99_ms": s["latency_p99_ms"],
                          "errors": s["errors"], "rss_growth_mb": s["rss_growth_mb"]})

    if not rows:
        print("No results found.")
        return

    fields = ["device", "model", "format", "batch", "status", "ms_per_img", "p99_call_ms", "img_per_s", "watts",
              "img_per_s_per_w", "ane_mw", "map_person", "map_all", "price_inr", "inr_per_img_s", "batch_ok", "error"]
    with open("compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def md_table(data, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for d in data:
            out.append("| " + " | ".join("" if d.get(c) is None else str(d.get(c)) for c in cols) + " |")
        return "\n".join(out)

    ok = [r for r in rows if r["status"] == "ok"]
    # Accuracy is measured on batch-1 exports only; copy it to the matching larger-batch rows.
    acc = {(r["device"], r["model"], r["format"]): (r["map_person"], r["map_all"]) for r in ok if r["batch"] == 1}
    for r in ok:
        if r["map_person"] is None and (r["device"], r["model"], r["format"]) in acc:
            r["map_person"], r["map_all"] = acc[(r["device"], r["model"], r["format"])]
    md = ["# YOLO26 hardware comparison", ""]

    # Best configuration per device and model (highest throughput, any batch)
    best = {}
    for r in ok:
        k = (r["device"], r["model"])
        if k not in best or r["img_per_s"] > best[k]["img_per_s"]:
            best[k] = r
    best_rows = sorted(best.values(), key=lambda r: (r["model"], -r["img_per_s"]))
    md += ["## Best configuration per device", "",
           md_table(best_rows, ["model", "device", "format", "batch", "img_per_s", "ms_per_img", "watts",
                                "img_per_s_per_w", "map_person", "inr_per_img_s"]), ""]

    md += ["## All runs", "",
           md_table(sorted(ok, key=lambda r: (r["model"], r["device"], r["format"], r["batch"])),
                    ["model", "device", "format", "batch", "ms_per_img", "p99_call_ms", "img_per_s", "watts",
                     "ane_mw", "map_person", "batch_ok"]), ""]

    if soaks:
        md += ["## Sustained run", "",
               md_table(soaks, ["device", "model", "format", "minutes", "fps_first", "fps_last", "drop_pct",
                                "p99_ms", "errors", "rss_growth_mb"]), ""]

    errs = [r for r in rows if r["status"] == "error"]
    if errs:
        md += ["## Failed runs", "", md_table(errs, ["device", "model", "format", "batch", "error"]), ""]

    Path("compare.md").write_text("\n".join(md))
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = sorted({r["model"] for r in best_rows})
        devices = sorted({r["device"] for r in best_rows})
        width = 0.8 / max(len(devices), 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, d in enumerate(devices):
            vals = [best.get((d, m), {}).get("img_per_s", 0) for m in models]
            ax.bar([x + i * width for x in range(len(models))], vals, width, label=d)
        ax.set_xticks([x + width * (len(devices) - 1) / 2 for x in range(len(models))])
        ax.set_xticklabels(models)
        ax.set_ylabel("Images per second (best config)")
        ax.set_title("YOLO26 throughput by device")
        ax.legend()
        fig.tight_layout()
        fig.savefig("compare_throughput.png", dpi=150)
        print("\nSaved compare.csv, compare.md, compare_throughput.png")
    except Exception as e:
        print(f"\nSaved compare.csv, compare.md (chart skipped: {e})")


if __name__ == "__main__":
    main()
