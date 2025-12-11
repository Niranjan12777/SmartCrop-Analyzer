import os
import numpy as np

PATCH_SIZE = 32
SAVE_PATCHES = True        # set False if you only want the S2-like cube
WAVELENGTHS_FILE = "indianpines_wavelengths.npy"   # optional; nm
HSI_FILE = "indianpinearray.npy"                   # (145,145,~200)
GT_FILE  = "IPgt.npy"                              # (145,145)

# Common Indian Pines water/noisy bands (indices in 0..B-1 space).

REMOVE_IDX = list(range(104,108)) + list(range(150,164))  # classic water bands

S2_SPECS = {
    "B1":  {"center": 442.7, "fwhm": 21.0},    # Coastal / Aerosol
    "B2":  {"center": 492.4, "fwhm": 66.0},    # Blue
    "B3":  {"center": 559.8, "fwhm": 36.0},    # Green
    "B4":  {"center": 664.6, "fwhm": 31.0},    # Red
    "B5":  {"center": 703.9, "fwhm": 16.0},    # RE1
    "B6":  {"center": 740.2, "fwhm": 15.0},    # RE2
    "B7":  {"center": 782.5, "fwhm": 20.0},    # RE3
    "B8":  {"center": 835.1, "fwhm":106.0},    # NIR (10 m)
    "B8A": {"center": 864.8, "fwhm": 21.0},    # NIR narrow
    "B9":  {"center": 945.0, "fwhm": 20.0},    # Water vapor
    "B10": {"center":1373.5, "fwhm": 30.0},    # Cirrus
    "B11": {"center":1613.7, "fwhm": 90.0},    # SWIR1
    "B12": {"center":2202.4, "fwhm":180.0},    # SWIR2
}
S2_ORDER = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

# -----------------------------
# 1) Helpers
# -----------------------------
def gaussian_srf(wavelengths_nm, center_nm, fwhm_nm):
    """Normalized Gaussian spectral response."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0*np.log(2.0)))
    w = np.exp(-0.5*((wavelengths_nm-center_nm)/sigma)**2)
    s = w.sum()
    return w / s if s > 0 else w

def synthesize_s2(hsi, wl_nm, remove_idx=None):
    """Create Sentinel-2 like bands by SRF-weighted integration."""
    H,W,B = hsi.shape
    if remove_idx:
        keep = np.array([i for i in range(B) if i not in set(remove_idx)], dtype=int)
        hsi = hsi[:,:,keep]
        wl_nm = wl_nm[keep]
        H,W,B = hsi.shape

    # per-band z-score normalize BEFORE integration to reduce sensor-scale bias
    mu = hsi.reshape(-1,B).mean(axis=0)
    sd = hsi.reshape(-1,B).std(axis=0) + 1e-8
    hsi_n = (hsi - mu) / sd

    out_bands = []
    valid_names = []
    for name in S2_ORDER:
        c = S2_SPECS[name]["center"]; f = S2_SPECS[name]["fwhm"]
        srf = gaussian_srf(wl_nm, c, f)               # (B,)
        if srf.sum() <= 0 or np.allclose(srf.sum(), 0):
            continue
        band = np.tensordot(hsi_n, srf, axes=([2],[0]))   # (H,W)
        out_bands.append(band)
        valid_names.append(name)

    out = np.stack(out_bands, axis=2)  # (H,W,N)
    # min-max scale each synthesized band to [0,1] for network compatibility
    mn = out.reshape(-1,out.shape[2]).min(axis=0)
    mx = out.reshape(-1,out.shape[2]).max(axis=0)
    out = (out - mn) / (mx - mn + 1e-8)
    return out, np.array(valid_names)

# -----------------------------
# 2) Load data & wavelengths
# -----------------------------
X = np.load(HSI_FILE)  # (145,145,B)
print("HSI shape:", X.shape)
if os.path.exists(WAVELENGTHS_FILE):
    wl_nm = np.load(WAVELENGTHS_FILE).astype(np.float64)  # (B,)
    if wl_nm.max() < 50:   # if given in microns, convert to nm
        wl_nm = wl_nm * 1000.0
    print("Loaded wavelengths from file:", wl_nm.shape)
else:
    # Fallback: linear 400–2500 nm across B bands
    B = X.shape[2]
    wl_nm = np.linspace(400.0, 2500.0, B)
    print("⚠ Using approximated wavelengths:", wl_nm.shape)

# -----------------------------
# 3) Synthesize Sentinel-2
# -----------------------------
X_s2, s2_names = synthesize_s2(X, wl_nm, remove_idx=REMOVE_IDX)
print("Synthesized S2-like cube:", X_s2.shape, "bands:", s2_names)

np.save("s2_sim.npy", X_s2)
print("Saved s2_sim.npy")

# -----------------------------
# 4) (Optional) Patch extraction aligned to IPgt
# -----------------------------
if SAVE_PATCHES:
    y = np.load(GT_FILE).astype(np.int32)   # (145,145)
    H,W,N = X_s2.shape
    assert y.shape[:2] == (H,W), "Label raster must match HSI spatial size."

    Xp, Yp = [], []
    for i in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
        for j in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
            x_patch = X_s2[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
            y_patch = y[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            # keep patches that contain any labeled pixels
            if np.any(y_patch > 0):
                Xp.append(x_patch)
                Yp.append(y_patch)

    Xp = np.stack(Xp, axis=0)
    Yp = np.stack(Yp, axis=0)
    np.save("unet_patches_s2.npy", Xp)
    np.save("unet_label_masks_s2.npy", Yp)
    print("Saved unet_patches_s2.npy:", Xp.shape, "unet_label_masks_s2.npy:", Yp.shape)
