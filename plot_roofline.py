#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def positive_float(value, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x) or x <= 0:
        return default
    return x


def nice_log_ticks(lo, hi):
    ticks = []
    start = math.floor(math.log10(lo))
    end = math.ceil(math.log10(hi))
    for exp in range(start, end + 1):
        for mant in (1, 2, 5):
            value = mant * (10**exp)
            if lo <= value <= hi:
                ticks.append(value)
    return ticks


def fmt_tick(value):
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def read_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ai = positive_float(row.get("ai_flop_per_byte"))
            custom = positive_float(row.get("custom_tflops"))
            cublas = positive_float(row.get("cublas_tflops"))
            peak = positive_float(row.get("compute_peak_tflops"))
            hbm = positive_float(row.get("hbm_tbps"))
            roof = positive_float(row.get("roofline_tflops"))
            if ai is None or peak is None or hbm is None:
                continue
            rows.append(
                {
                    "label": f"{row.get('M')}x{row.get('N')}x{row.get('K')}",
                    "ai": ai,
                    "custom": custom,
                    "cublas": cublas,
                    "peak": peak,
                    "hbm": hbm,
                    "roof": roof,
                    "bound": row.get("roofline_bound", ""),
                }
            )
    if not rows:
        raise SystemExit(f"No roofline rows found in {path}")
    return rows


def polyline(points):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def main():
    parser = argparse.ArgumentParser(description="Generate an SVG roofline plot from gemm_bench CSV.")
    parser.add_argument("--csv", required=True, help="CSV from ./gemm_bench --csv")
    parser.add_argument("--out", default="roofline.svg", help="Output SVG path")
    parser.add_argument("--title", default="GEMM Roofline")
    args = parser.parse_args()

    rows = read_rows(args.csv)
    peak = rows[0]["peak"]
    hbm = rows[0]["hbm"]
    ridge = peak / hbm

    ais = [r["ai"] for r in rows] + [ridge]
    ys = [r["roof"] for r in rows if r["roof"] is not None] + [peak]
    ys += [r["custom"] for r in rows if r["custom"] is not None]
    ys += [r["cublas"] for r in rows if r["cublas"] is not None]

    x_min = min(ais) / 4.0
    x_max = max(ais) * 4.0
    y_min = max(min(ys) / 4.0, 1.0e-4)
    y_max = max(ys) * 2.0

    x_min = 10 ** math.floor(math.log10(x_min))
    x_max = 10 ** math.ceil(math.log10(x_max))
    y_min = 10 ** math.floor(math.log10(y_min))
    y_max = 10 ** math.ceil(math.log10(y_max))

    width = 1100
    height = 760
    left = 92
    right = 40
    top = 60
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x):
        return left + (math.log10(x) - math.log10(x_min)) / (
            math.log10(x_max) - math.log10(x_min)
        ) * plot_w

    def sy(y):
        return top + (math.log10(y_max) - math.log10(y)) / (
            math.log10(y_max) - math.log10(y_min)
        ) * plot_h

    xs = []
    exp_start = math.floor(math.log10(x_min))
    exp_end = math.ceil(math.log10(x_max))
    for exp in range(exp_start, exp_end + 1):
        for mant in (1, 1.5, 2, 3, 5, 7):
            x = mant * (10**exp)
            if x_min <= x <= x_max:
                xs.append(x)
    xs = sorted(set(xs + [ridge]))

    roof_points = [(sx(x), sy(min(peak, hbm * x))) for x in xs]
    mem_points = [(sx(x), sy(hbm * x)) for x in xs if hbm * x <= peak]
    compute_points = [(sx(x), sy(peak)) for x in xs if x >= ridge]

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" '
        f'font-family="Arial" font-size="22" font-weight="700">{svg_escape(args.title)}</text>'
    )
    parts.append(
        f'<text x="{width / 2:.0f}" y="54" text-anchor="middle" '
        f'font-family="Arial" font-size="13" fill="#555">'
        f'peak={peak:.3f} TFLOPS, HBM={hbm:.3f} TB/s, ridge AI={ridge:.3f} FLOPs/byte</text>'
    )

    # Grid and ticks.
    for x in nice_log_ticks(x_min, x_max):
        px = sx(x)
        major = abs(math.log10(x) - round(math.log10(x))) < 1.0e-9
        color = "#d9d9d9" if major else "#eeeeee"
        parts.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="{color}" stroke-width="1"/>')
        if major:
            parts.append(
                f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
                f'font-family="Arial" font-size="12" fill="#444">{fmt_tick(x)}</text>'
            )

    for y in nice_log_ticks(y_min, y_max):
        py = sy(y)
        major = abs(math.log10(y) - round(math.log10(y))) < 1.0e-9
        color = "#d9d9d9" if major else "#eeeeee"
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="{color}" stroke-width="1"/>')
        if major:
            parts.append(
                f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end" '
                f'font-family="Arial" font-size="12" fill="#444">{fmt_tick(y)}</text>'
            )

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.5"/>')

    # Roofline.
    if mem_points:
        parts.append(f'<polyline points="{polyline(mem_points)}" fill="none" stroke="#4c78a8" stroke-width="3"/>')
    if compute_points:
        parts.append(f'<polyline points="{polyline(compute_points)}" fill="none" stroke="#f58518" stroke-width="3"/>')
    parts.append(f'<polyline points="{polyline(roof_points)}" fill="none" stroke="#222" stroke-width="2" stroke-dasharray="7,5"/>')
    parts.append(
        f'<line x1="{sx(ridge):.2f}" y1="{top}" x2="{sx(ridge):.2f}" y2="{top + plot_h}" '
        f'stroke="#888" stroke-width="1.5" stroke-dasharray="4,5"/>'
    )
    parts.append(
        f'<text x="{sx(ridge) + 6:.2f}" y="{top + 16}" font-family="Arial" '
        f'font-size="12" fill="#555">ridge</text>'
    )

    # Data points.
    for r in rows:
        x = sx(r["ai"])
        if r["custom"] is not None:
            y = sy(r["custom"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="#e45756" stroke="#8c1d18" stroke-width="1"/>')
            parts.append(f'<title>custom {svg_escape(r["label"])}: AI={r["ai"]:.3f}, TF={r["custom"]:.3f}</title>')
        if r["cublas"] is not None:
            y = sy(r["cublas"])
            parts.append(f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" height="10" fill="#54a24b" stroke="#1d5e1f" stroke-width="1"/>')

    # Labels for custom points.
    for r in rows:
        if r["custom"] is None:
            continue
        x = sx(r["ai"])
        y = sy(r["custom"])
        parts.append(
            f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="Arial" '
            f'font-size="11" fill="#333">{svg_escape(r["label"])}</text>'
        )

    # Axis labels.
    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 26}" text-anchor="middle" '
        f'font-family="Arial" font-size="16">Arithmetic intensity (FLOPs / byte)</text>'
    )
    parts.append(
        f'<text transform="translate(24 {top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle" '
        f'font-family="Arial" font-size="16">Performance (TFLOPS)</text>'
    )

    # Legend.
    lx = left + plot_w - 210
    ly = top + 24
    parts.append(f'<rect x="{lx - 14}" y="{ly - 20}" width="210" height="98" fill="white" stroke="#ccc"/>')
    parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 30}" y2="{ly}" stroke="#4c78a8" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 38}" y="{ly + 4}" font-family="Arial" font-size="12">memory roof</text>')
    parts.append(f'<line x1="{lx}" y1="{ly + 22}" x2="{lx + 30}" y2="{ly + 22}" stroke="#f58518" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 38}" y="{ly + 26}" font-family="Arial" font-size="12">compute roof</text>')
    parts.append(f'<circle cx="{lx + 15}" cy="{ly + 44}" r="5.5" fill="#e45756" stroke="#8c1d18" stroke-width="1"/>')
    parts.append(f'<text x="{lx + 38}" y="{ly + 48}" font-family="Arial" font-size="12">custom kernel</text>')
    parts.append(f'<rect x="{lx + 10}" y="{ly + 61}" width="10" height="10" fill="#54a24b" stroke="#1d5e1f" stroke-width="1"/>')
    parts.append(f'<text x="{lx + 38}" y="{ly + 70}" font-family="Arial" font-size="12">cuBLAS</text>')

    parts.append("</svg>")

    out = Path(args.out)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

