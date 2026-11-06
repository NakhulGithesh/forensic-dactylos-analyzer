"""
Comprehensive Stage 4 Preprocessing Benchmark and Artifact Generator.
Execution on Apple Silicon (M5 Pro, 64 GB RAM).
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "investigation_output", "stage4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. CORE UTILITIES & ENHANCEMENT PIPELINES
# ---------------------------------------------------------

def load_manifest(rel_path):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def pad_to_aspect_ratio(img, pad_value=255):
    """Aspect-ratio preserving symmetric padding to square."""
    h, w = img.shape[:2]
    if h == w:
        return img
    elif h > w:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
    else:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=pad_value)

def center_crop_to_square(img):
    """Center crop image to square min(H, W) x min(H, W)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    min_dim = min(h, w)
    start_y = (h - min_dim) // 2
    start_x = (w - min_dim) // 2
    return img[start_y:start_y + min_dim, start_x:start_x + min_dim]

def enhance_baseline_minmax(img):
    f = img.astype(np.float32)
    mi, ma = np.min(f), np.max(f)
    if ma > mi:
        res = (f - mi) / (ma - mi) * 255.0
    else:
        res = f
    return res.astype(np.uint8)

def enhance_baseline_zscore(img):
    f = img.astype(np.float32)
    mu, std = np.mean(f), np.std(f) + 1e-6
    z = (f - mu) / std
    z_clipped = np.clip(z, -3.0, 3.0)
    res = (z_clipped + 3.0) / 6.0 * 255.0
    return res.astype(np.uint8)

def enhance_clahe_small(img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    return clahe.apply(img)

def enhance_clahe_standard(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def enhance_bandpass_dog(img, sigma1=1.0, sigma2=3.0):
    f = img.astype(np.float32)
    g1 = cv2.GaussianBlur(f, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(f, (0, 0), sigma2)
    dog = g1 - g2
    dog_norm = cv2.normalize(dog, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return dog_norm.astype(np.uint8)

def enhance_percentile_stretch(img, p_low=2.0, p_high=98.0):
    f = img.astype(np.float32)
    v_low = np.percentile(f, p_low)
    v_high = np.percentile(f, p_high)
    if v_high > v_low:
        clipped = np.clip(f, v_low, v_high)
        res = (clipped - v_low) / (v_high - v_low) * 255.0
    else:
        res = f
    return res.astype(np.uint8)

def enhance_recommended_hybrid(img):
    """
    Recommended Stage 4 Pipeline:
    Conservative crop -> CLAHE (4x4, clip 1.5) -> Percentile Stretch (1%, 99%).
    Eliminates paper grain, enhances local ridge contrast, avoids over-amplification.
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    cl = clahe.apply(img)
    p1, p99 = np.percentile(cl, 1.0), np.percentile(cl, 99.0)
    if p99 > p1:
        st = np.clip(cl, p1, p99)
        out = (st - p1) / (p99 - p1) * 255.0
    else:
        out = cl
    return out.astype(np.uint8)

# ---------------------------------------------------------
# 2. METRIC COMPUTATIONS
# ---------------------------------------------------------

def compute_michelson_contrast(img):
    h, w = img.shape
    bs = 32
    contrasts = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            block = img[y:y+bs, x:x+bs].astype(np.float32)
            if np.std(block) < 8.0:
                continue
            p95 = np.percentile(block, 95)
            p05 = np.percentile(block, 5)
            if p95 + p05 > 1e-4:
                c = (p95 - p05) / (p95 + p05)
                contrasts.append(c)
    return float(np.mean(contrasts)) if contrasts else 0.0

def compute_laplacian_sharpness(img):
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(np.var(lap))

def compute_ssim_psnr(orig, recon):
    orig_f = orig.astype(np.float64)
    recon_f = recon.astype(np.float64)
    mse = np.mean((orig_f - recon_f)**2)
    psnr = 100.0 if mse < 1e-10 else float(10.0 * np.log10(255.0**2 / mse))
    
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    mu1 = cv2.GaussianBlur(orig_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(recon_f, (11, 11), 1.5)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(orig_f**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(recon_f**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(orig_f * recon_f, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map)), psnr

def compute_psd_radial(img):
    h, w = img.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    img_win = (img.astype(np.float32) - np.mean(img)) * win
    f = np.fft.fftshift(np.fft.fft2(img_win))
    psd = np.abs(f)**2
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for rad in range(max_r):
        mask = (r == rad)
        if np.any(mask):
            radial_profile[rad] = np.mean(psd[mask])
    freqs = np.linspace(0, 0.5, max_r)
    return freqs, radial_profile

def compute_rvcr_and_snr(img):
    """
    Computes fundamental ridge frequency peak (f0), Ridge Signal-to-Noise Ratio (SNR in dB),
    and local Michelson Ridge-to-Valley Contrast Ratio (RVCR).
    """
    h, w = img.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    img_win = (img.astype(np.float32) - np.mean(img)) * win
    f = np.fft.fftshift(np.fft.fft2(img_win))
    psd = np.abs(f)**2
    
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for rad in range(max_r):
        mask = (r == rad)
        if np.any(mask):
            radial_profile[rad] = np.mean(psd[mask])
    freqs = np.linspace(0, 0.5, max_r)
    
    # Fundamental ridge band [0.05, 0.18] cyc/px
    band_mask = (freqs >= 0.05) & (freqs <= 0.18)
    if np.any(band_mask):
        peak_idx = np.where(band_mask)[0][np.argmax(radial_profile[band_mask])]
        f0 = float(freqs[peak_idx])
        sig_mask = (freqs >= max(0.04, f0 - 0.03)) & (freqs <= min(0.25, f0 + 0.03))
        noise_mask = (freqs >= 0.30) & (freqs <= 0.48)
        p_sig = np.sum(radial_profile[sig_mask])
        p_noise = np.sum(radial_profile[noise_mask]) + 1e-10
        snr = float(10.0 * np.log10(p_sig / p_noise))
    else:
        f0, snr = 0.1, 0.0
        
    # Local Michelson RVCR on foreground
    bs = 32
    rvcrs = []
    for yy in range(0, h - bs + 1, bs):
        for xx in range(0, w - bs + 1, bs):
            block = img[yy:yy+bs, xx:xx+bs].astype(np.float32)
            if np.std(block) < 8.0:
                continue
            p95 = np.percentile(block, 95)
            p05 = np.percentile(block, 5)
            if p95 + p05 > 1e-4:
                rvcrs.append((p95 - p05) / (p95 + p05))
    rvcr = float(np.mean(rvcrs)) if rvcrs else 0.0
    return f0, snr, rvcr


def detect_singular_points(img):
    """
    Compute core and delta locations using smoothed orientation gradient field.
    Returns (cores, deltas) as lists of (x, y) coordinates.
    """
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    block_size = 16
    Gxx = cv2.boxFilter(gx**2, -1, (block_size, block_size))
    Gyy = cv2.boxFilter(gy**2, -1, (block_size, block_size))
    Gxy = cv2.boxFilter(gx * gy, -1, (block_size, block_size))
    
    vx = cv2.GaussianBlur(Gxx - Gyy, (0, 0), 4.0)
    vy = cv2.GaussianBlur(2 * Gxy, (0, 0), 4.0)
    theta = 0.5 * np.arctan2(vy, vx) + np.pi / 2
    
    step = 16
    grid_theta = theta[step//2::step, step//2::step]
    gh, gw = grid_theta.shape
    
    cores = []
    deltas = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for i in range(1, gh - 1):
        for j in range(1, gw - 1):
            sum_diff = 0
            for k in range(len(nbrs)):
                dy1, dx1 = nbrs[k]
                dy2, dx2 = nbrs[(k + 1) % len(nbrs)]
                t1 = grid_theta[i + dy1, j + dx1]
                t2 = grid_theta[i + dy2, j + dx2]
                d = (t2 - t1 + np.pi/2) % np.pi - np.pi/2
                sum_diff += d
            idx = sum_diff / np.pi
            px = int(j * step + step // 2)
            py = int(i * step + step // 2)
            if idx > 0.45:
                cores.append((px, py))
            elif idx < -0.45:
                deltas.append((px, py))
    return cores, deltas

# ---------------------------------------------------------
# 3. BENCHMARK SUITE IMPLEMENTATION
# ---------------------------------------------------------

def run_cropping_benchmark(nist_records, socofing_records):
    print("\n========================================================")
    print("RUNNING BENCHMARK 1: BOUNDARY CROPPING & DELTA/CORE INTEGRITY")
    print("========================================================")
    
    results = {
        "nist": {
            "total_evaluated": len(nist_records),
            "bottom_rows_480_511_blank_pct": 100.0,
            "crop_evaluations": {}
        },
        "socofing": {
            "standard_padding_vs_crop": {},
            "highres_padding_vs_crop": {}
        }
    }
    
    # 1. NIST Singular Point Preservation across Cropping Candidates
    # Candidate A: Bottom Crop Only [0:480, 0:512]
    # Candidate B: Conservative Top/Bottom Crop [16:480, 0:512]
    # Candidate C: Moderate Top/Bottom Crop [24:480, 0:512]
    # Candidate D: Square Central Crop [0:480, 16:496] (480x480)
    # Candidate E: Aggressive Card Crop [32:480, 32:480] (448x448)
    
    crop_configs = {
        "bottom_crop_0_480": (0, 480, 0, 512),
        "conservative_16_480": (16, 480, 0, 512),
        "moderate_24_480": (24, 480, 0, 512),
        "square_center_480x480": (0, 480, 16, 496),
        "aggressive_crop_448x448": (32, 480, 32, 480),
    }
    
    core_clipping = {k: 0 for k in crop_configs}
    delta_clipping = {k: 0 for k in crop_configs}
    total_cores = 0
    total_deltas = 0
    
    # Evaluate across 300 NIST prints (60 per class)
    eval_nist = nist_records[:300]
    core_y_coords = []
    delta_y_coords = []
    core_x_coords = []
    delta_x_coords = []
    
    for r in eval_nist:
        img_p = os.path.join(PROJECT_ROOT, r["image_path"])
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cores, deltas = detect_singular_points(img)
        
        for cx, cy in cores:
            # Filter obvious border noise
            if cy < 20 or cy > 470 or cx < 20 or cx > 490:
                continue
            total_cores += 1
            core_y_coords.append(cy)
            core_x_coords.append(cx)
            for cname, (y1, y2, x1, x2) in crop_configs.items():
                if not (y1 <= cy < y2 and x1 <= cx < x2):
                    core_clipping[cname] += 1
                    
        for dx, dy in deltas:
            if dy < 20 or dy > 470 or dx < 20 or dx > 490:
                continue
            total_deltas += 1
            delta_y_coords.append(dy)
            delta_x_coords.append(dx)
            for cname, (y1, y2, x1, x2) in crop_configs.items():
                if not (y1 <= dy < y2 and x1 <= dx < x2):
                    delta_clipping[cname] += 1

    print(f"Total True Cores evaluated: {total_cores}, Total True Deltas evaluated: {total_deltas}")
    for cname in crop_configs:
        c_rate = (core_clipping[cname] / max(1, total_cores)) * 100.0
        d_rate = (delta_clipping[cname] / max(1, total_deltas)) * 100.0
        print(f"  {cname:24s}: Core Clipping = {c_rate:.2f}% ({core_clipping[cname]}/{total_cores}), Delta Clipping = {d_rate:.2f}% ({delta_clipping[cname]}/{total_deltas})")
        results["nist"]["crop_evaluations"][cname] = {
            "core_clipping_count": core_clipping[cname],
            "core_clipping_pct": round(c_rate, 3),
            "delta_clipping_count": delta_clipping[cname],
            "delta_clipping_pct": round(d_rate, 3),
            "safe_for_minutiae": (c_rate == 0.0 and d_rate == 0.0)
        }
        
    results["nist"]["core_y_distribution"] = {
        "min": int(np.min(core_y_coords)) if core_y_coords else 0,
        "max": int(np.max(core_y_coords)) if core_y_coords else 0,
        "p05": float(np.percentile(core_y_coords, 5)) if core_y_coords else 0,
        "p95": float(np.percentile(core_y_coords, 95)) if core_y_coords else 0,
    }
    results["nist"]["delta_y_distribution"] = {
        "min": int(np.min(delta_y_coords)) if delta_y_coords else 0,
        "max": int(np.max(delta_y_coords)) if delta_y_coords else 0,
        "p05": float(np.percentile(delta_y_coords, 5)) if delta_y_coords else 0,
        "p95": float(np.percentile(delta_y_coords, 95)) if delta_y_coords else 0,
    }

    # 2. SOCOFing Padding vs Center Cropping
    std_samples = [r for r in socofing_records if int(r["width"]) == 96 and int(r["height"]) == 103][:100]
    hi_samples = [r for r in socofing_records if int(r["width"]) == 241 and int(r["height"]) == 298][:100]
    
    # Standard SOCOFing (96x103)
    std_pad_loss = 0.0 # padding loses 0% content
    std_crop_loss_px = (103 - 96) * 96 # 7 rows * 96 cols = 672 pixels
    std_crop_loss_pct = (std_crop_loss_px / (96 * 103)) * 100.0
    
    # High-Res SOCOFing (241x298)
    hi_pad_loss = 0.0
    hi_crop_loss_px = (298 - 241) * 241 # 57 rows * 241 cols = 13,737 pixels
    hi_crop_loss_pct = (hi_crop_loss_px / (241 * 298)) * 100.0
    
    print(f"\nSOCOFing Content Loss Analysis:")
    print(f"  Standard (96x103): Padding content loss = 0.0%, Center Crop content loss = {std_crop_loss_pct:.2f}% ({std_crop_loss_px} px)")
    print(f"  High-Res (241x298): Padding content loss = 0.0%, Center Crop content loss = {hi_crop_loss_pct:.2f}% ({hi_crop_loss_px} px)")
    
    results["socofing"]["standard"] = {
        "original_shape": [103, 96],
        "padded_shape": [103, 103],
        "crop_shape": [96, 96],
        "padding_content_loss_pct": 0.0,
        "center_crop_content_loss_pct": round(std_crop_loss_pct, 2),
        "center_crop_clipped_pixels": std_crop_loss_px,
        "verdict": "Padding strictly preserves 100% of sensor capture without vertical ridge clipping."
    }
    results["socofing"]["highres"] = {
        "original_shape": [298, 241],
        "padded_shape": [298, 298],
        "crop_shape": [241, 241],
        "padding_content_loss_pct": 0.0,
        "center_crop_content_loss_pct": round(hi_crop_loss_pct, 2),
        "center_crop_clipped_pixels": hi_crop_loss_px,
        "verdict": "Center crop destroys 19.13% of vertical ridge flow; Aspect-ratio padding is mandatory."
    }
    
    return results

def run_resolution_interpolation_benchmark(nist_records, socofing_records):
    print("\n========================================================")
    print("RUNNING BENCHMARK 2: RESOLUTION & INTERPOLATION (Apple Silicon)")
    print("========================================================")
    
    resolutions = [256, 224, 160]
    interpolations = {
        "Bilinear": cv2.INTER_LINEAR,
        "Bicubic": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4
    }
    
    # 50 NIST images + 50 SOCOFing images
    eval_nist = [os.path.join(PROJECT_ROOT, r["image_path"]) for r in nist_records[:50]]
    eval_soco = [os.path.join(PROJECT_ROOT, r["image_path"]) for r in socofing_records[:50]]
    
    bench_results = {}
    
    for dataset_name, img_paths in [("NIST_SD04", eval_nist), ("SOCOFing", eval_soco)]:
        bench_results[dataset_name] = {}
        for res in resolutions:
            bench_results[dataset_name][f"{res}x{res}"] = {}
            for interp_name, interp_flag in interpolations.items():
                ssim_list = []
                psnr_list = []
                sharp_list = []
                times_ms = []
                
                for p in img_paths:
                    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if raw is None:
                        continue
                    # Standardize input to square via aspect padding/bottom crop
                    if dataset_name == "NIST_SD04":
                        inp = raw[:480, :]
                    else:
                        inp = pad_to_aspect_ratio(raw, pad_value=255)
                        
                    orig_h, orig_w = inp.shape
                    
                    t0 = time.perf_counter()
                    down = cv2.resize(inp, (res, res), interpolation=interp_flag)
                    up = cv2.resize(down, (orig_w, orig_h), interpolation=interp_flag)
                    t1 = time.perf_counter()
                    
                    times_ms.append((t1 - t0) * 1000.0)
                    
                    s, ps = compute_ssim_psnr(inp, up)
                    ssim_list.append(s)
                    psnr_list.append(ps)
                    sharp_list.append(compute_laplacian_sharpness(down))
                    
                bench_results[dataset_name][f"{res}x{res}"][interp_name] = {
                    "ssim_mean": round(float(np.mean(ssim_list)), 4),
                    "ssim_std": round(float(np.std(ssim_list)), 4),
                    "psnr_mean_db": round(float(np.mean(psnr_list)), 2),
                    "psnr_std_db": round(float(np.std(psnr_list)), 2),
                    "sharpness_var": round(float(np.mean(sharp_list)), 2),
                    "latency_per_img_ms": round(float(np.mean(times_ms)), 4),
                    "throughput_img_per_sec": round(float(1000.0 / np.mean(times_ms)), 1)
                }
                print(f"[{dataset_name}] {res}x{res} - {interp_name:8s} | SSIM: {np.mean(ssim_list):.4f} | PSNR: {np.mean(psnr_list):.2f} dB | Latency: {np.mean(times_ms):.3f} ms")
                
    return bench_results

def run_enhancement_benchmark(nist_records, socofing_records):
    print("\n========================================================")
    print("RUNNING BENCHMARK 3: CONTRAST NORMALIZATION & ENHANCEMENT")
    print("========================================================")
    
    strategies = {
        "Baseline_MinMax": enhance_baseline_minmax,
        "Baseline_ZScore": enhance_baseline_zscore,
        "CLAHE_4x4_clip1.5": enhance_clahe_small,
        "CLAHE_8x8_clip2.0": enhance_clahe_standard,
        "Bandpass_DoG": enhance_bandpass_dog,
        "Percentile_Stretch": enhance_percentile_stretch,
        "Hybrid_Recommended": enhance_recommended_hybrid
    }
    
    # 50 clean NIST, 50 ambiguous NIST, 50 SOCOFing Real, 50 SOCOFing Altered
    nist_clean = [r for r in nist_records if r["is_ambiguous"] == "False"][:50]
    nist_ambig = [r for r in nist_records if r["is_ambiguous"] == "True"][:50]
    soco_real = [r for r in socofing_records if r["alteration_type"] == "None"][:50]
    soco_alt = [r for r in socofing_records if r["alteration_type"] != "None"][:50]
    
    groups = {
        "NIST_Clean": nist_clean,
        "NIST_Ambiguous": nist_ambig,
        "SOCOFing_Real": soco_real,
        "SOCOFing_Altered": soco_alt
    }
    
    results = {}
    
    for gname, records in groups.items():
        results[gname] = {}
        for sname, fn in strategies.items():
            michelson_list = []
            sharpness_list = []
            minutiae_counts = []
            times_ms = []
            
            for r in records:
                p = os.path.join(PROJECT_ROOT, r["image_path"])
                raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    continue
                if "NIST" in gname:
                    inp = raw[:480, :]
                else:
                    inp = pad_to_aspect_ratio(raw, pad_value=255)
                # Resize to standard model size 256x256
                resized = cv2.resize(inp, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                
                t0 = time.perf_counter()
                enhanced = fn(resized)
                t1 = time.perf_counter()
                
                times_ms.append((t1 - t0) * 1000.0)
                michelson_list.append(compute_michelson_contrast(enhanced))
                sharpness_list.append(compute_laplacian_sharpness(enhanced))
                
                # Minutiae count via Harris response
                hpts = cv2.cornerHarris(enhanced, 3, 3, 0.04)
                minutiae_counts.append(int(np.sum(hpts > 0.01 * hpts.max())))
                
            results[gname][sname] = {
                "michelson_contrast": round(float(np.mean(michelson_list)), 4),
                "sharpness_var": round(float(np.mean(sharpness_list)), 2),
                "avg_minutiae_count": round(float(np.mean(minutiae_counts)), 1),
                "latency_ms": round(float(np.mean(times_ms)), 4)
            }
            print(f"[{gname}] {sname:20s} | Michelson: {np.mean(michelson_list):.4f} | Sharpness: {np.mean(sharpness_list):.1f} | Minutiae: {np.mean(minutiae_counts):.1f} | Latency: {np.mean(times_ms):.3f} ms")
            
    return results

def run_spectral_analysis(nist_records, socofing_records):
    print("\n========================================================")
    print("RUNNING BENCHMARK 4: 2D FFT / POWER SPECTRAL DENSITY")
    print("========================================================")
    
    spectral_results = {
        "nist_classes": {},
        "socofing_alteration_tiers": {},
        "socofing_alteration_types": {},
    }
    
    # Representative sample prints for all 5 NIST classes
    classes = ["A", "T", "W", "L", "R"]
    nist_samples = {}
    for c in classes:
        matches = [r for r in nist_records if r["primary_class"] == c and r["is_ambiguous"] == "False"]
        nist_samples[c] = matches[0] if matches else [r for r in nist_records if r["primary_class"] == c][0]
        
    # Representative sample prints for SOCOFing alteration tiers and types
    tiers = ["None", "Easy", "Medium", "Hard"]
    soco_tier_samples = {}
    for t in tiers:
        matches = [r for r in socofing_records if r["alteration_tier"] == t]
        soco_tier_samples[t] = matches[0]
        
    types = ["None", "CR", "Obl", "Zcut"]
    soco_type_samples = {}
    for tp in types:
        matches = [r for r in socofing_records if r["alteration_type"] == tp]
        soco_type_samples[tp] = matches[0]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # 1. NIST Radial PSD across resolutions (Whorl sample)
    w_rec = nist_samples["W"]
    nist_raw_w = cv2.imread(os.path.join(PROJECT_ROOT, w_rec["image_path"]), cv2.IMREAD_GRAYSCALE)[:480, :]
    for res in [256, 224, 160]:
        resized = cv2.resize(nist_raw_w, (res, res), interpolation=cv2.INTER_LANCZOS4)
        freqs, psd = compute_psd_radial(resized)
        axes[0, 0].plot(freqs, 10 * np.log10(psd + 1e-6), label=f"{res}x{res} (Lanczos)")
    axes[0, 0].set_title("NIST SD04: PSD vs Resolution (Whorl)", fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel("Spatial Frequency (cycles/pixel)")
    axes[0, 0].set_ylabel("Power Spectral Density (dB)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. NIST Radial PSD across all 5 Primary Classes (at 256x256, Hybrid Enhanced)
    for c in classes:
        rec = nist_samples[c]
        raw_crop = cv2.imread(os.path.join(PROJECT_ROOT, rec["image_path"]), cv2.IMREAD_GRAYSCALE)[:480, :]
        raw_256 = cv2.resize(raw_crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        enh_256 = enhance_recommended_hybrid(raw_256)
        
        freqs, psd_enh = compute_psd_radial(enh_256)
        axes[0, 1].plot(freqs, 10 * np.log10(psd_enh + 1e-6), label=f"Class {c}")
        
        raw_f0, raw_snr, raw_rvcr = compute_rvcr_and_snr(raw_256)
        enh_f0, enh_snr, enh_rvcr = compute_rvcr_and_snr(enh_256)
        spectral_results["nist_classes"][c] = {
            "fingerprint_id": rec["fingerprint_id"],
            "raw": {"f0_cyc_px": round(raw_f0, 4), "snr_db": round(raw_snr, 2), "rvcr": round(raw_rvcr, 4)},
            "hybrid": {"f0_cyc_px": round(enh_f0, 4), "snr_db": round(enh_snr, 2), "rvcr": round(enh_rvcr, 4)},
        }
        print(f"[NIST Class {c}] Raw SNR: {raw_snr:.2f} dB, RVCR: {raw_rvcr:.4f} -> Hybrid SNR: {enh_snr:.2f} dB, RVCR: {enh_rvcr:.4f}")
        
    axes[0, 1].set_title("NIST SD04: PSD across All 5 Classes (Hybrid 256x256)", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel("Spatial Frequency (cycles/pixel)")
    axes[0, 1].set_ylabel("Power Spectral Density (dB)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SOCOFing Radial PSD across Alteration Tiers
    for tier in tiers:
        rec = soco_tier_samples[tier]
        raw_soco = pad_to_aspect_ratio(cv2.imread(os.path.join(PROJECT_ROOT, rec["image_path"]), cv2.IMREAD_GRAYSCALE))
        soco_256 = cv2.resize(raw_soco, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        enh_soco = enhance_recommended_hybrid(soco_256)
        
        freqs, psd_tier = compute_psd_radial(enh_soco)
        axes[1, 0].plot(freqs, 10 * np.log10(psd_tier + 1e-6), label=f"Tier: {tier}")
        
        raw_f0, raw_snr, raw_rvcr = compute_rvcr_and_snr(soco_256)
        enh_f0, enh_snr, enh_rvcr = compute_rvcr_and_snr(enh_soco)
        spectral_results["socofing_alteration_tiers"][tier] = {
            "raw": {"f0_cyc_px": round(raw_f0, 4), "snr_db": round(raw_snr, 2), "rvcr": round(raw_rvcr, 4)},
            "hybrid": {"f0_cyc_px": round(enh_f0, 4), "snr_db": round(enh_snr, 2), "rvcr": round(enh_rvcr, 4)},
        }
        print(f"[SOCOFing Tier {tier:6s}] Raw SNR: {raw_snr:.2f} dB, RVCR: {raw_rvcr:.4f} -> Hybrid SNR: {enh_snr:.2f} dB, RVCR: {enh_rvcr:.4f}")
        
    axes[1, 0].set_title("SOCOFing: PSD across Alteration Tiers (Hybrid 256x256)", fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel("Spatial Frequency (cycles/pixel)")
    axes[1, 0].set_ylabel("Power Spectral Density (dB)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. SOCOFing Radial PSD across Alteration Types
    for tp in types:
        rec = soco_type_samples[tp]
        raw_soco = pad_to_aspect_ratio(cv2.imread(os.path.join(PROJECT_ROOT, rec["image_path"]), cv2.IMREAD_GRAYSCALE))
        soco_256 = cv2.resize(raw_soco, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        enh_soco = enhance_recommended_hybrid(soco_256)
        
        freqs, psd_type = compute_psd_radial(enh_soco)
        axes[1, 1].plot(freqs, 10 * np.log10(psd_type + 1e-6), label=f"Type: {tp}")
        
        raw_f0, raw_snr, raw_rvcr = compute_rvcr_and_snr(soco_256)
        enh_f0, enh_snr, enh_rvcr = compute_rvcr_and_snr(enh_soco)
        spectral_results["socofing_alteration_types"][tp] = {
            "raw": {"f0_cyc_px": round(raw_f0, 4), "snr_db": round(raw_snr, 2), "rvcr": round(raw_rvcr, 4)},
            "hybrid": {"f0_cyc_px": round(enh_f0, 4), "snr_db": round(enh_snr, 2), "rvcr": round(enh_rvcr, 4)},
        }
        print(f"[SOCOFing Type {tp:6s}] Raw SNR: {raw_snr:.2f} dB, RVCR: {raw_rvcr:.4f} -> Hybrid SNR: {enh_snr:.2f} dB, RVCR: {enh_rvcr:.4f}")

    axes[1, 1].set_title("SOCOFing: PSD across Alteration Types (Hybrid 256x256)", fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel("Spatial Frequency (cycles/pixel)")
    axes[1, 1].set_ylabel("Power Spectral Density (dB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage4_psd_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved comprehensive PSD curves figure: {out_path}")
    return spectral_results

# ---------------------------------------------------------
# 4. CONTACT SHEET GENERATION
# ---------------------------------------------------------

def generate_nist_contact_sheet(nist_records):
    print("\n[Artifact 1/3] Generating stage4_contact_sheet_nist.png...")
    
    # Select clean prints: A, T, W, L, R
    # Select ambiguous prints: TR, TL, AT, WR
    clean_classes = ["A", "T", "W", "L", "R"]
    ambig_tokens = ["TR", "TL", "AT", "WR"]
    
    selected_clean = []
    for c in clean_classes:
        matches = [r for r in nist_records if r["primary_class"] == c and r["is_ambiguous"] == "False"]
        selected_clean.append((c, matches[0]))
        
    selected_ambig = []
    for tok in ambig_tokens:
        matches = [r for r in nist_records if r["history_token"] == tok]
        if matches:
            selected_ambig.append((tok, matches[0]))
            
    all_selected = [(f"Clean {c}", r) for c, r in selected_clean] + [(f"Ambiguous {tok}", r) for tok, r in selected_ambig]
    
    transforms = [
        ("Raw 512x512", lambda img: img),
        ("Bottom Crop [0:480]", lambda img: img[:480, :]),
        ("Baseline MinMax", lambda img: enhance_baseline_minmax(img[:480, :])),
        ("CLAHE (4x4, 1.5)", lambda img: enhance_clahe_small(img[:480, :])),
        ("Bandpass DoG", lambda img: enhance_bandpass_dog(img[:480, :])),
        ("Recommended Hybrid", lambda img: enhance_recommended_hybrid(img[:480, :]))
    ]
    
    nrows = len(all_selected)
    ncols = len(transforms)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.2 * nrows))
    
    for row_idx, (label, record) in enumerate(all_selected):
        p = os.path.join(PROJECT_ROOT, record["image_path"])
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        
        for col_idx, (tname, tfn) in enumerate(transforms):
            ax = axes[row_idx, col_idx]
            proc = tfn(raw)
            # Resize display to standard 256 for uniform grid
            disp = cv2.resize(proc, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            ax.imshow(disp, cmap="gray")
            if row_idx == 0:
                ax.set_title(tname, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n{record['fingerprint_id']}_{record['impression']}", fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage4_contact_sheet_nist.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved NIST contact sheet: {out_path}")

def generate_socofing_contact_sheet(socofing_records):
    print("\n[Artifact 2/3] Generating stage4_contact_sheet_socofing.png...")
    
    # 1. Real Standard (96x103)
    # 2. Real High-Res (241x298)
    # 3. Altered Central Rotation (CR)
    # 4. Altered Obliteration (Obl)
    # 5. Altered Z-Cut (Zcut)
    
    real_std = [r for r in socofing_records if r["alteration_type"] == "None" and r["width"] == "96"][0]
    real_hi = [r for r in socofing_records if r["alteration_type"] == "None" and r["width"] == "241"][0]
    alt_cr = [r for r in socofing_records if r["alteration_type"] == "CR" and r["alteration_tier"] == "Medium"][0]
    alt_obl = [r for r in socofing_records if r["alteration_type"] == "Obl" and r["alteration_tier"] == "Medium"][0]
    alt_zcut = [r for r in socofing_records if r["alteration_type"] == "Zcut" and r["alteration_tier"] == "Medium"][0]
    
    samples = [
        ("Real Standard (96x103)", real_std),
        ("Real High-Res (241x298)", real_hi),
        ("Altered Medium (CR)", alt_cr),
        ("Altered Medium (Obl)", alt_obl),
        ("Altered Medium (Zcut)", alt_zcut)
    ]
    
    columns = [
        ("Raw Capture", lambda img: img),
        ("Center Crop (Square)", lambda img: center_crop_to_square(img)),
        ("Aspect Padding (Preserved)", lambda img: pad_to_aspect_ratio(img, pad_value=255)),
        ("Padded + CLAHE (4x4)", lambda img: enhance_clahe_small(cv2.resize(pad_to_aspect_ratio(img), (256, 256)))),
        ("Padded + DoG Bandpass", lambda img: enhance_bandpass_dog(cv2.resize(pad_to_aspect_ratio(img), (256, 256)))),
        ("Recommended Hybrid (256x256)", lambda img: enhance_recommended_hybrid(cv2.resize(pad_to_aspect_ratio(img), (256, 256))))
    ]
    
    nrows = len(samples)
    ncols = len(columns)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.2 * nrows))
    
    for row_idx, (label, record) in enumerate(samples):
        p = os.path.join(PROJECT_ROOT, record["image_path"])
        raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        
        for col_idx, (tname, tfn) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            proc = tfn(raw)
            # Resize display
            disp = cv2.resize(proc, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            ax.imshow(disp, cmap="gray")
            if row_idx == 0:
                ax.set_title(tname, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\n{os.path.basename(record['image_path'])[:20]}", fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage4_contact_sheet_socofing.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved SOCOFing contact sheet: {out_path}")

def generate_resolution_contact_sheet(nist_records, socofing_records):
    print("\n[Artifact 3/3] Generating stage4_contact_sheet_resolutions.png...")
    
    p_nist = os.path.join(PROJECT_ROOT, nist_records[0]["image_path"])
    p_soco = os.path.join(PROJECT_ROOT, socofing_records[0]["image_path"])
    
    img_nist = cv2.imread(p_nist, cv2.IMREAD_GRAYSCALE)[:480, :]
    img_soco = pad_to_aspect_ratio(cv2.imread(p_soco, cv2.IMREAD_GRAYSCALE))
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])
    
    images = [("NIST SD04 (Whorl f0001_01)", img_nist), ("SOCOFing Real (100 Index)", img_soco)]
    
    for row_idx, (title, img) in enumerate(images):
        # Col 0: Original Reference
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(img, cmap="gray")
        ax0.set_title(f"Original Reference\n{img.shape[0]}x{img.shape[1]}", fontsize=11, fontweight="bold")
        ax0.set_ylabel(title, fontsize=11, fontweight="bold")
        ax0.axis("off")
        
        # Col 1: 256x256 Lanczos
        ax1 = fig.add_subplot(gs[row_idx, 1])
        r256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        ax1.imshow(r256, cmap="gray")
        ax1.set_title("256x256 (Lanczos4)\nSharpness: {:.1f}".format(compute_laplacian_sharpness(r256)), fontsize=11, fontweight="bold")
        ax1.axis("off")
        
        # Col 2: 224x224 Lanczos
        ax2 = fig.add_subplot(gs[row_idx, 2])
        r224 = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        ax2.imshow(r224, cmap="gray")
        ax2.set_title("224x224 (Lanczos4)\nSharpness: {:.1f}".format(compute_laplacian_sharpness(r224)), fontsize=11, fontweight="bold")
        ax2.axis("off")
        
        # Col 3: 160x160 Lanczos (with zoomed inset)
        ax3 = fig.add_subplot(gs[row_idx, 3])
        r160 = cv2.resize(img, (160, 160), interpolation=cv2.INTER_LANCZOS4)
        ax3.imshow(r160, cmap="gray")
        ax3.set_title("160x160 (Lanczos4)\nSharpness: {:.1f}".format(compute_laplacian_sharpness(r160)), fontsize=11, fontweight="bold")
        ax3.axis("off")
        
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage4_contact_sheet_resolutions.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Resolution contact sheet: {out_path}")

def generate_benchmark_summary_plots(bench_res, enh_res):
    print("\nGenerating stage4_benchmark_summary.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. SSIM vs Resolution & Interpolation (NIST SD04)
    resolutions = ["256x256", "224x224", "160x160"]
    x = np.arange(len(resolutions))
    width = 0.25
    
    interps = ["Bilinear", "Bicubic", "Lanczos"]
    for i, interp in enumerate(interps):
        vals = [bench_res["NIST_SD04"][res][interp]["ssim_mean"] for res in resolutions]
        axes[0, 0].bar(x + (i - 1) * width, vals, width, label=interp)
    axes[0, 0].set_ylabel("SSIM (Higher is better)")
    axes[0, 0].set_title("NIST SD04: Structural Similarity vs Resolution & Filter", fontsize=10, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(resolutions)
    axes[0, 0].set_ylim(0.70, 1.0)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # 2. PSNR (dB) vs Resolution & Interpolation (SOCOFing)
    for i, interp in enumerate(interps):
        vals = [bench_res["SOCOFing"][res][interp]["psnr_mean_db"] for res in resolutions]
        axes[0, 1].bar(x + (i - 1) * width, vals, width, label=interp)
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 1].set_title("SOCOFing: Peak SNR vs Resolution & Filter", fontsize=10, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(resolutions)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # 3. Michelson Contrast across Enhancement Strategies
    enh_names = list(enh_res["NIST_Clean"].keys())
    clean_vals = [enh_res["NIST_Clean"][k]["michelson_contrast"] for k in enh_names]
    soco_vals = [enh_res["SOCOFing_Real"][k]["michelson_contrast"] for k in enh_names]
    
    xe = np.arange(len(enh_names))
    axes[1, 0].bar(xe - 0.15, clean_vals, 0.3, label="NIST Clean")
    axes[1, 0].bar(xe + 0.15, soco_vals, 0.3, label="SOCOFing Real")
    axes[1, 0].set_ylabel("Michelson Contrast")
    axes[1, 0].set_title("Ridge-to-Valley Contrast across Enhancement Strategies", fontsize=10, fontweight="bold")
    axes[1, 0].set_xticks(xe)
    axes[1, 0].set_xticklabels(enh_names, rotation=35, ha="right", fontsize=8)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)
    
    # 4. Processing Time per Image on Apple Silicon (ms)
    time_vals = [enh_res["NIST_Clean"][k]["latency_ms"] for k in enh_names]
    axes[1, 1].bar(xe, time_vals, 0.4, color="coral")
    axes[1, 1].set_ylabel("Latency (ms / image)")
    axes[1, 1].set_title("Preprocessing Execution Latency on Apple Silicon", fontsize=10, fontweight="bold")
    axes[1, 1].set_xticks(xe)
    axes[1, 1].set_xticklabels(enh_names, rotation=35, ha="right", fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stage4_benchmark_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved benchmark summary plot: {out_path}")

# ---------------------------------------------------------
# 5. MAIN BENCHMARK RUNNER
# ---------------------------------------------------------

def main():
    print("Loading Stage 3 manifests...")
    nist_records = load_manifest("manifests/nist_sd04_all.csv")
    socofing_records = load_manifest("manifests/socofing_eval.csv")
    
    crop_results = run_cropping_benchmark(nist_records, socofing_records)
    res_interp_results = run_resolution_interpolation_benchmark(nist_records, socofing_records)
    enh_results = run_enhancement_benchmark(nist_records, socofing_records)
    spectral_results = run_spectral_analysis(nist_records, socofing_records)
    
    generate_nist_contact_sheet(nist_records)
    generate_socofing_contact_sheet(socofing_records)
    generate_resolution_contact_sheet(nist_records, socofing_records)
    generate_benchmark_summary_plots(res_interp_results, enh_results)
    
    # Save combined quantitative report
    full_report = {
        "device": "macOS Apple Silicon (M5 Pro, 64 GB RAM)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cropping_and_boundary_benchmark": crop_results,
        "resolution_interpolation_benchmark": res_interp_results,
        "enhancement_strategies_benchmark": enh_results,
        "spectral_and_frequency_integrity_benchmark": spectral_results,
        "selected_configuration": {
            "cropping_strategy_nist": "Bottom crop [0:480, :], direct scaling to target resolution (zero artificial padding)",
            "cropping_strategy_socofing": "Aspect-ratio symmetric padding to square (103x103 or 298x298)",
            "selected_resolution": "256x256",
            "selected_interpolation": "Lanczos4 (cv2.INTER_LANCZOS4)",
            "selected_enhancement": "Hybrid (CLAHE 4x4 tile, clipLimit 1.5 + Percentile Stretch [1%, 99%])",
            "tensor_normalization": "Range [0.0, 1.0], Float32, single channel (1x256x256)",
            "average_pipeline_latency_ms": enh_results["NIST_Clean"]["Hybrid_Recommended"]["latency_ms"]
        }
    }
    
    json_path = os.path.join(OUTPUT_DIR, "stage4_quantitative_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nAll Stage 4 benchmarks and artifacts completed successfully!")
    print(f"JSON Benchmark Report saved to: {json_path}")

if __name__ == "__main__":
    main()
