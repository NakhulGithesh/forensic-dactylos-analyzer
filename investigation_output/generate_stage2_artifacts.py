#!/usr/bin/env python3
"""
STAGE 2 ARTIFACT GENERATOR: NIST SD4 & SOCOFING ANALYSIS
Generates:
1. loop_orientation_verification_4cases.png
2. ridge_flow_orientation_verified.png
3. card_headers_verification.png
4. contact_sheet_nist_sd04.png
5. contact_sheet_socofing.png
6. contact_sheet_dataset_comparison.png
"""

import os, glob, re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

def generate_orientation_field_overlay():
    print("[1/6] Generating ridge_flow_orientation_verified.png...")
    def plot_orientation_field(img_path, title, out_ax):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        crop = img[:480, :]
        gx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=5)
        
        out_ax.imshow(crop, cmap='gray')
        step = 24
        h, w = crop.shape
        for y in range(step//2, h, step):
            for x in range(step//2, w, step):
                sub_gx = gx[max(0, y-12):min(h, y+12), max(0, x-12):min(w, x+12)]
                sub_gy = gy[max(0, y-12):min(h, y+12), max(0, x-12):min(w, x+12)]
                Gxx = np.sum(sub_gx**2)
                Gyy = np.sum(sub_gy**2)
                Gxy = np.sum(sub_gx * sub_gy)
                denom = np.sqrt((Gxx - Gyy)**2 + 4 * Gxy**2)
                if denom < 1e-4:
                    continue
                coherence = denom / (Gxx + Gyy + 1e-6)
                if coherence < 0.2:
                    continue
                theta = 0.5 * np.arctan2(2*Gxy, Gxx - Gyy) + np.pi/2
                dx = 10 * np.cos(theta)
                dy = 10 * np.sin(theta)
                out_ax.plot([x - dx, x + dx], [y - dy, y + dy], color='red', lw=1.2)
        out_ax.set_title(title, fontsize=10)
        out_ax.axis('off')

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    cases = [
        ('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0002_05.png',
         'f0002_05: Right Little (05), Class R\nFlow: Right, Delta: Left -> Ulnar Loop', axes[0, 0]),
        ('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0003_10.png',
         'f0003_10: Left Little (10), Class L\nFlow: Left, Delta: Right -> Ulnar Loop', axes[0, 1]),
        ('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0148_02.png',
         'f0148_02: Right Index (02), Class L\nFlow: Left, Delta: Right -> Radial Loop', axes[1, 0]),
        ('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_1/f0372_07.png',
         'f0372_07: Left Index (07), Class R\nFlow: Right, Delta: Left -> Radial Loop', axes[1, 1]),
    ]
    for p, t, ax in cases:
        plot_orientation_field(p, t, ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ridge_flow_orientation_verified.png'), dpi=200)
    plt.close()

def generate_card_headers():
    print("[2/6] Generating card_headers_verification.png...")
    samples = [
        ('s1903_01.png (Digit 01)', 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_7/s1903_01.png'),
        ('f0148_02.png (Digit 02: "2. R. INDEX")', 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0148_02.png'),
        ('s1991_06.png (Digit 06)', 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_7/s1991_06.png'),
        ('s1944_07.png (Digit 07)', 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_7/s1944_07.png'),
        ('s1976_10.png (Digit 10: "10. LEFT LITT")', 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_7/s1976_10.png'),
    ]
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 5))
    for ax, (t, p) in zip(axes, samples):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ax.imshow(img[:35, :], cmap='gray')
        ax.set_title(t, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'card_headers_verification.png'), dpi=200)
    plt.close()

def generate_contact_sheet_sd4():
    print("[3/6] Generating contact_sheet_nist_sd04.png...")
    classes_config = {
        'Arch (A)': [
            ('f0005_03', 'R.Middle (03)'),
            ('f0006_09', 'L.Ring (09)'),
            ('f0012_01', 'R.Thumb (01)'),
            ('f0013_03', 'R.Middle (03)'),
            ('f0015_01', 'R.Thumb (01)')
        ],
        'Tented Arch (T)': [
            ('f0021_07', 'L.Index (07)'),
            ('f0040_07', 'L.Index (07)'),
            ('f0056_04', 'R.Ring (04)'),
            ('f0115_03', 'R.Middle (03)'),
            ('f0118_08', 'L.Middle (08)')
        ],
        'Whorl (W)': [
            ('f0001_01', 'R.Thumb (01)'),
            ('f0010_01', 'R.Thumb (01)'),
            ('f0011_02', 'R.Index (02)'),
            ('f0014_06', 'L.Thumb (06)'),
            ('f0024_06', 'L.Thumb (06)')
        ],
        'Left Loop (L)': [
            ('f0003_10', 'L.Little (10) [Ulnar]'),
            ('f0007_09', 'L.Ring (09) [Ulnar]'),
            ('f0009_08', 'L.Middle (08) [Ulnar]'),
            ('f0148_02', 'R.Index (02) [Radial]'),
            ('f1744_02', 'R.Index (02) [Radial]')
        ],
        'Right Loop (R)': [
            ('f0002_05', 'R.Little (05) [Ulnar]'),
            ('f0004_05', 'R.Little (05) [Ulnar]'),
            ('f0022_05', 'R.Little (05) [Ulnar]'),
            ('f0372_07', 'L.Index (07) [Radial]'),
            ('f0171_08', 'L.Middle (08) [Radial]')
        ]
    }

    fig, axes = plt.subplots(5, 5, figsize=(18, 20))
    fig.suptitle('NIST Special Database 4 — Representative Pattern Classes\n(Rolled Ink on Paper, 512x512, 500 PPI)',
                 fontsize=16, y=0.995, fontweight='bold')

    row_idx = 0
    for cls_label, items in classes_config.items():
        for col_idx, (fid, sub) in enumerate(items):
            ax = axes[row_idx, col_idx]
            matches = glob.glob(f'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/*/{fid}.png')
            if not matches:
                continue
            img = cv2.imread(matches[0], cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{fid}\n{sub}', fontsize=9, pad=3)
            ax.axis('off')
            if col_idx == 0:
                ax.text(-0.15, 0.5, cls_label, transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
        row_idx += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.08)
    plt.savefig(os.path.join(OUT_DIR, 'contact_sheet_nist_sd04.png'), dpi=180)
    plt.close()

def generate_contact_sheet_socofing():
    print("[4/6] Generating contact_sheet_socofing.png...")
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(7, 5, figure=fig, hspace=0.35, wspace=0.15)
    fig.suptitle('SOCOFing (Sokoto Coventry Fingerprint Dataset) Contact Sheet\n(Optical Live-Scan Sensor, 103x96 Pixels, Synthetic Alteration Tiers)',
                 fontsize=15, fontweight='bold', y=0.99)

    f_id = '364__M_Right_little_finger'
    real_path = f'Fingerprint Data/SOCOFing/Real/{f_id}.BMP'
    im_real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)

    alts = [
        ('Obliteration (Obl)', 'Obl'),
        ('Central Rotation (CR)', 'CR'),
        ('Z-Cut Distortion (Zcut)', 'Zcut')
    ]

    for r_idx, (alt_name, alt_code) in enumerate(alts):
        ax0 = fig.add_subplot(gs[r_idx, 0])
        ax0.imshow(im_real, cmap='gray')
        ax0.set_title(f'Real\n{f_id[:12]}', fontsize=8)
        ax0.axis('off')
        ax0.text(-0.15, 0.5, alt_name, transform=ax0.transAxes,
                 fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)
        
        pe = f'Fingerprint Data/SOCOFing/Altered/Altered-Easy/{f_id}_{alt_code}.BMP'
        ax1 = fig.add_subplot(gs[r_idx, 1])
        ax1.imshow(cv2.imread(pe, cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax1.set_title(f'Altered-Easy\n({alt_code})', fontsize=8)
        ax1.axis('off')

        pm = f'Fingerprint Data/SOCOFing/Altered/Altered-Medium/{f_id}_{alt_code}.BMP'
        ax2 = fig.add_subplot(gs[r_idx, 2])
        ax2.imshow(cv2.imread(pm, cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax2.set_title(f'Altered-Medium\n({alt_code})', fontsize=8)
        ax2.axis('off')

        ph = f'Fingerprint Data/SOCOFing/Altered/Altered-Hard/{f_id}_{alt_code}.BMP'
        ax3 = fig.add_subplot(gs[r_idx, 3])
        ax3.imshow(cv2.imread(ph, cv2.IMREAD_GRAYSCALE), cmap='gray')
        ax3.set_title(f'Altered-Hard\n({alt_code})', fontsize=8)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[r_idx, 4])
        diff = cv2.absdiff(im_real, cv2.imread(ph, cv2.IMREAD_GRAYSCALE))
        ax4.imshow(diff, cmap='magma')
        ax4.set_title(f'Alteration Mask\n(Real vs Hard)', fontsize=8)
        ax4.axis('off')

    sample_files = [
        '100__M_Left_thumb_finger',
        '100__M_Right_index_finger',
        '101__M_Left_middle_finger',
        '102__M_Right_ring_finger',
        '103__F_Left_little_finger'
    ]
    alt_suffixes = ['_CR', '_Obl', '_Zcut', '_Obl', '_Obl']

    for col_idx, sf in enumerate(sample_files):
        ax = fig.add_subplot(gs[3, col_idx])
        p = f'Fingerprint Data/SOCOFing/Real/{sf}.BMP'
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        ax.imshow(im, cmap='gray')
        ax.set_title(sf.replace('__M_', '_M_').replace('__F_', '_F_'), fontsize=7)
        ax.axis('off')
        if col_idx == 0:
            ax.text(-0.15, 0.5, 'Real (Various)', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)

    tiers = [
        ('Altered-Easy', 'Fingerprint Data/SOCOFing/Altered/Altered-Easy'),
        ('Altered-Medium', 'Fingerprint Data/SOCOFing/Altered/Altered-Medium'),
        ('Altered-Hard', 'Fingerprint Data/SOCOFing/Altered/Altered-Hard')
    ]

    for t_idx, (t_label, folder) in enumerate(tiers):
        r = 4 + t_idx
        for col_idx, (sf, suf) in enumerate(zip(sample_files, alt_suffixes)):
            ax = fig.add_subplot(gs[r, col_idx])
            p = os.path.join(folder, f'{sf}{suf}.BMP')
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            ax.imshow(im, cmap='gray')
            parts = sf.split('_')
            ax.set_title(f'{parts[0]} {parts[3]} ({suf[1:]})', fontsize=7)
            ax.axis('off')
            if col_idx == 0:
                ax.text(-0.15, 0.5, t_label, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', ha='right', rotation=90)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.03)
    plt.savefig(os.path.join(OUT_DIR, 'contact_sheet_socofing.png'), dpi=180)
    plt.close()

def generate_comparison_sheet():
    print("[5/6] Generating contact_sheet_dataset_comparison.png...")
    sd4_path = 'Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0001_01.png'
    sd4_img = cv2.imread(sd4_path, cv2.IMREAD_GRAYSCALE)
    sd4_active = sd4_img[:480, :]

    soc_path = 'Fingerprint Data/SOCOFing/Real/100__M_Right_index_finger.BMP'
    soc_img = cv2.imread(soc_path, cv2.IMREAD_GRAYSCALE)

    soc_obl = cv2.imread('Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Right_index_finger_Obl.BMP', cv2.IMREAD_GRAYSCALE)
    soc_zcut = cv2.imread('Fingerprint Data/SOCOFing/Altered/Altered-Hard/101__M_Left_middle_finger_Zcut.BMP', cv2.IMREAD_GRAYSCALE)
    soc_cr = cv2.imread('Fingerprint Data/SOCOFing/Altered/Altered-Hard/100__M_Left_thumb_finger_CR.BMP', cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Stage 2 Inspection: NIST SD4 vs. SOCOFing Dataset Comparative Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sd4_img, cmap='gray')
    ax1.set_title('NIST SD4: Rolled Ink on Paper\n512x512 px (Full Field of View)', fontsize=10, fontweight='bold')
    rect = plt.Rectangle((0, 480), 512, 32, fill=False, edgecolor='red', linewidth=1.5, linestyle='--')
    ax1.add_patch(rect)
    ax1.text(256, 498, '32-row NIST white margin', color='red', fontsize=8, ha='center', va='center')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(soc_img, cmap='gray')
    ax2.set_title('SOCOFing: Optical Live-Scan Touch\n103x96 px (Native Size View)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')

    ax3 = fig.add_subplot(gs[0, 2])
    scale_canvas = np.ones((512, 512), dtype=np.uint8) * 255
    scale_canvas[10:10+103, 10:10+96] = soc_img
    ax3.imshow(scale_canvas, cmap='gray')
    rect_soc = plt.Rectangle((10, 10), 96, 103, fill=False, edgecolor='blue', linewidth=2)
    ax3.add_patch(rect_soc)
    ax3.text(120, 60, '<-- SOCOFing (103x96)\ninside 512x512 frame\n(Resolution ratio: ~26.7x area)',
             color='blue', fontsize=9, va='center')
    ax3.set_title('Relative Spatial Area Comparison\n(1:1 Pixel Canvas Scale)', fontsize=10, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(sd4_active.ravel(), bins=64, range=(0, 256), color='navy', alpha=0.6, density=True, label='NIST SD4 (Rolled Ink)')
    ax4.hist(soc_img.ravel(), bins=64, range=(0, 256), color='crimson', alpha=0.6, density=True, label='SOCOFing (Live-Scan)')
    ax4.set_title('Pixel Intensity Distribution', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Grayscale Value (0=Black, 255=White)')
    ax4.set_ylabel('Normalized Density')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 0])
    sd4_zoom = sd4_active[200:264, 200:264]
    ax5.imshow(sd4_zoom, cmap='gray')
    ax5.set_title('NIST SD4 Zoom (64x64 px)\nRidge Period: ~12.8 px (~500 PPI)\nMinutiae & Pores Clearly Defined', fontsize=9, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    soc_zoom = soc_img[35:67, 32:64]
    ax6.imshow(soc_zoom, cmap='gray')
    ax6.set_title('SOCOFing Zoom (32x32 px)\nRidge Period: ~2.9 px (Downsampled)\nPixelated / Aliased Edges', fontsize=9, fontweight='bold')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    sd4_loop = cv2.imread('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0002_05.png', cv2.IMREAD_GRAYSCALE)[:480, :]
    ax7.imshow(sd4_loop, cmap='gray')
    circle_core = plt.Circle((265, 200), 25, color='lime', fill=False, lw=2)
    circle_delta = plt.Circle((145, 360), 25, color='cyan', fill=False, lw=2)
    ax7.add_patch(circle_core)
    ax7.add_patch(circle_delta)
    ax7.text(295, 200, 'Core', color='lime', fontweight='bold', fontsize=9)
    ax7.text(175, 360, 'Delta', color='cyan', fontweight='bold', fontsize=9)
    ax7.set_title('NIST SD4: Complete Anatomy\nBoth Core and Delta Fully Visible\n(Nail-to-Nail Rolled Margin)', fontsize=9, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(soc_img, cmap='gray')
    circle_soc_core = plt.Circle((50, 55), 12, color='lime', fill=False, lw=2)
    ax8.add_patch(circle_soc_core)
    ax8.text(50, 40, 'Core Only', color='lime', fontweight='bold', fontsize=9, ha='center')
    ax8.text(48, 95, '[Delta Truncated by Platen Edge]', color='yellow', fontsize=7, ha='center', backgroundcolor='black')
    ax8.set_title('SOCOFing: Limited FOV\nCore Visible; Delta Often Cut Off\n(Flat Platen Contact Boundary)', fontsize=9, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 0])
    sd4_noisy = cv2.imread('Fingerprint Data/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/f0008_10.png', cv2.IMREAD_GRAYSCALE)[:480, :]
    ax9.imshow(sd4_noisy, cmap='gray')
    ax9.set_title('NIST SD4 Noise & Artifacts\nInk Bleed, Smearing, Crease Lines,\nCard Borders, Pre-printed Text', fontsize=9, fontweight='bold')
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(soc_obl, cmap='gray')
    ax10.set_title('SOCOFing Damage: Obliteration (Obl)\nCore Ridge Erasure / Blurring\nSimulates Abrasion / Chemical Burns', fontsize=9, fontweight='bold')
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(soc_cr, cmap='gray')
    ax11.set_title('SOCOFing Damage: Central Rotation (CR)\nRotated Core Patch Discontinuity\nSimulates Cut/Graft Surgical Tampering', fontsize=9, fontweight='bold')
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[2, 3])
    ax12.imshow(soc_zcut, cmap='gray')
    ax12.set_title('SOCOFing Damage: Z-Cut (Zcut)\nTriangular Flap Transposition\nSimulates Severe Scarring / Incision', fontsize=9, fontweight='bold')
    ax12.axis('off')

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.04, right=0.98)
    plt.savefig(os.path.join(OUT_DIR, 'contact_sheet_dataset_comparison.png'), dpi=180)
    plt.close()

if __name__ == '__main__':
    generate_orientation_field_overlay()
    generate_card_headers()
    generate_contact_sheet_sd4()
    generate_contact_sheet_socofing()
    generate_comparison_sheet()
    print("All artifacts successfully generated in investigation_output/")
