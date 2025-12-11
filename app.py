# prediction_service/app.py
import os
import re
import zipfile
import tempfile
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib
matplotlib.use("Agg")  # MUST be before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory

# CONFIG
MODEL_PATH = os.path.join("model_files", "unet_crop_analysis.h5")
EXPECTED_BANDS = 13
PATCH_SIZE = 32
STRIDE = PATCH_SIZE // 3
OUTPUT_ROOT = os.path.join(os.getcwd(), "Outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

app = Flask(__name__)

# -------------------------------------------------------
# Band token helpers (detect B01..B12, B8A)
# -------------------------------------------------------
BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
]
band_token_re = re.compile(r"(B0?\d{1,2}A?|B8A)", flags=re.IGNORECASE)

def extract_band_token(filename):
    m = band_token_re.search(filename)
    if not m:
        return None
    token = m.group(0).upper()
    # normalize tokens like B8 -> B08 (but keep B8A)
    if token == "B8":
        token = "B08"
    if re.fullmatch(r"B\d$", token):
        token = "B0" + token[1:]
    return token

# -------------------------------------------------------
# Load bands from folder with robust ordering
# -------------------------------------------------------
def load_bands_from_folder(folder, expected_bands=EXPECTED_BANDS):
    tif_files = [f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))]
    if not tif_files:
        raise FileNotFoundError("No .tif found in folder")

    token_map = {}
    files_without_token = []
    for f in tif_files:
        tok = extract_band_token(f)
        if tok:
            token_map[f] = tok
        else:
            files_without_token.append(f)

    ordered_files = []
    if token_map:
        token_to_files = {}
        for fname, tok in token_map.items():
            token_to_files.setdefault(tok, []).append(fname)
        for tok in BAND_ORDER:
            if tok in token_to_files:
                for fname in sorted(token_to_files[tok]):
                    ordered_files.append(fname)
        # append any other detected tokens (rare)
        other_tokens = sorted([t for t in set(token_map.values()) if t not in BAND_ORDER])
        for tok in other_tokens:
            for fname in sorted([f for f, tt in token_map.items() if tt == tok]):
                ordered_files.append(fname)
        ordered_files.extend(sorted(files_without_token))
    else:
        ordered_files = sorted(tif_files)

    print("Using band file order:")
    for f in ordered_files:
        print("  ", f)

    bands = []
    meta = None
    for f in ordered_files:
        fp = os.path.join(folder, f)
        with rasterio.open(fp) as src:
            bands.append(src.read(1))
            if meta is None:
                meta = src.meta.copy()

    bands_array = np.stack(bands, axis=0)
    available_bands = bands_array.shape[0]

    if available_bands < expected_bands:
        pad_count = expected_bands - available_bands
        pad = np.zeros((pad_count, bands_array.shape[1], bands_array.shape[2]), dtype=bands_array.dtype)
        bands_array = np.concatenate([bands_array, pad], axis=0)
        print(f"Padded missing {pad_count} bands to match {expected_bands}.")
    elif available_bands > expected_bands:
        bands_array = bands_array[:expected_bands]
        print(f"Trimmed to {expected_bands} bands to match model input.")

    return bands_array

# -------------------------------------------------------
# Utility: convert raw band array to float reflectance 0..1
# - If max value looks like scaled reflectance (<=10000), divide by 10000
# - Else, scale each band by its percentile range (1-99) to reduce outliers
# -------------------------------------------------------
def scale_to_reflectance(bands_array):
    # bands_array: shape (bands, H, W)
    arr = bands_array.astype(np.float32)
    global_max = np.nanmax(arr)
    global_min = np.nanmin(arr)
    print(f"Raw bands min={global_min:.3f} max={global_max:.3f} dtype={bands_array.dtype}")

    if global_max > 1.0 and global_max <= 10000.0:
        print("Detected integer reflectance-like data; dividing by 10000.")
        arr = arr / 10000.0
        arr = np.clip(arr, 0.0, 1.0)
    else:
        # per-band robust scaling using 1st-99th percentiles
        bands = arr.shape[0]
        for b in range(bands):
            band = arr[b]
            p1 = np.percentile(band, 1)
            p99 = np.percentile(band, 99)
            if p99 - p1 > 0:
                arr[b] = (band - p1) / (p99 - p1)
            else:
                # fallback to simple min-max
                mn = np.nanmin(band); mx = np.nanmax(band)
                if mx - mn > 0:
                    arr[b] = (band - mn) / (mx - mn)
                else:
                    arr[b] = 0.0
        arr = np.clip(arr, 0.0, 1.0)
    return arr

# -------------------------------------------------------
# Core pipeline: prediction + NDRE + NDMI computation
# -------------------------------------------------------
def generate_maps_from_array(bands_array, user_id):
    # Input shape: (bands, H, W)
    # Convert to HWC and scale
    bands_array = bands_array.astype(np.float32)
    scaled = scale_to_reflectance(bands_array)
    img = reshape_as_image(scaled)  # (H, W, C)
    H, W, C = img.shape
    print(f"Prepared image shape (H,W,C) = {img.shape}")

    # Load model
    model = load_model(MODEL_PATH)

    num_classes = 17
    pred_probs = np.zeros((H, W, num_classes), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    print("Running overlapping patch prediction...")
    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):
            patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
            patch = np.expand_dims(patch, 0)
            pred = model.predict(patch, verbose=0)[0]
            pred_probs[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += pred
            count_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += 1

    count_map[count_map == 0] = 1
    pred_probs /= count_map[..., None]

    # Health map: max confidence
    health_map = np.max(pred_probs, axis=-1)

    # --- NDRE (stress) ---
    # Use indices assuming bands are ordered as B01..B12 with B8->index7, B05->index4
    try:
        nir = img[:, :, 7].astype(np.float32)   # B08
        red_edge = img[:, :, 4].astype(np.float32)  # B05
    except IndexError:
        raise RuntimeError("NDRE bands not found — check band ordering / expected bands.")

    ndre = (nir - red_edge) / (nir + red_edge + 1e-9)
    ndre = np.clip(ndre, -1.0, 1.0)
    # map -1..1 to 0..1 (safer than global min/max)
    ndre_norm = (ndre + 1.0) / 2.0
    stress_map = 1.0 - ndre_norm

    # Debug stats
    print(f"NDRE stats: min={ndre.min():.4f}, max={ndre.max():.4f}, median={np.median(ndre):.4f}")
    print(f"NDRE_norm stats: min={ndre_norm.min():.4f}, max={ndre_norm.max():.4f}")

    # --- NDMI (moisture) using B8A (index 8) and B11 (index 10)
    try:
        nir_a = img[:, :, 8].astype(np.float32)  # B8A
        swir1 = img[:, :, 10].astype(np.float32)  # B11
    except IndexError:
        raise RuntimeError("NDMI bands not found — check band ordering / expected bands.")

    ndmi = (nir_a - swir1) / (nir_a + swir1 + 1e-9)
    ndmi = np.clip(ndmi, -1.0, 1.0)
    ndmi_norm = (ndmi + 1.0) / 2.0

    print(f"NDMI stats: min={ndmi.min():.4f}, max={ndmi.max():.4f}, median={np.median(ndmi):.4f}")

    # Save results per user
    user_out = os.path.join(OUTPUT_ROOT, str(user_id))
    os.makedirs(user_out, exist_ok=True)

    def save_map(arr, cmap, title, fname):
        path = os.path.join(user_out, fname)
        plt.figure(figsize=(8,6))
        plt.imshow(arr, cmap=cmap)
        plt.colorbar(label=title)
        plt.title(title)
        plt.axis("off")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    health_path = save_map(health_map, "RdYlGn", "Crop Health Map", "health_map.png")
    stress_path = save_map(stress_map, "Reds", "Crop Stress Map (NDRE)", "stress_map.png")
    moisture_cmap = LinearSegmentedColormap.from_list("moisture", ["red","orange","white","blue"])
    moisture_path = save_map(ndmi_norm, moisture_cmap, "Crop Moisture Map (NDMI)", "moisture_map.png")

    return {"health": health_path, "stress": stress_path, "moisture": moisture_path}

# -------------------------------------------------------
# /predict endpoint — supports multiple files, ZIP, or single TIFF
# -------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    user_id = request.form.get("user_id", "anonymous")

    # Case: multiple separate tif files uploaded as "files"
    if "files" in request.files:
        tmpdir = tempfile.mkdtemp(prefix="pred_")
        for f in request.files.getlist("files"):
            fp = os.path.join(tmpdir, f.filename)
            f.save(fp)
        bands_array = load_bands_from_folder(tmpdir)
        result = generate_maps_from_array(bands_array, user_id)
        return jsonify(result), 200

    # Case: single file uploaded as 'file' (zip or multiband tiff)
    file = request.files.get("file", None)
    if file is None:
        return jsonify({"error": "No file provided"}), 400

    tmpdir = tempfile.mkdtemp(prefix="pred_")
    fp = os.path.join(tmpdir, file.filename)
    file.save(fp)

    if file.filename.lower().endswith(".zip"):
        with zipfile.ZipFile(fp, 'r') as z:
            z.extractall(tmpdir)
        bands_array = load_bands_from_folder(tmpdir)
    elif file.filename.lower().endswith((".tif", ".tiff")):
        with rasterio.open(fp) as src:
            arr = src.read()  # (bands, H, W)
        if arr.shape[0] < EXPECTED_BANDS:
            pad = np.zeros((EXPECTED_BANDS - arr.shape[0], arr.shape[1], arr.shape[2]))
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > EXPECTED_BANDS:
            arr = arr[:EXPECTED_BANDS]
        bands_array = arr
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    result = generate_maps_from_array(bands_array, user_id)
    return jsonify(result), 200

# -------------------------------------------------------
# predict_local and outputs endpoints (unchanged)
# -------------------------------------------------------
@app.route("/predict_local", methods=["POST"])
def predict_local():
    body = request.get_json(force=True)
    path = body.get("path")
    user_id = body.get("user_id", "local")
    if not path or not os.path.exists(path):
        return jsonify({"error": "path not found"}), 400
    if os.path.isdir(path):
        bands_array = load_bands_from_folder(path)
    else:
        if not path.lower().endswith((".tif", ".tiff")):
            return jsonify({"error": "unsupported"}), 400
        with rasterio.open(path) as src:
            arr = src.read()
        if arr.shape[0] < EXPECTED_BANDS:
            pad = np.zeros((EXPECTED_BANDS - arr.shape[0], arr.shape[1], arr.shape[2]))
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > EXPECTED_BANDS:
            arr = arr[:EXPECTED_BANDS]
        bands_array = arr
    result = generate_maps_from_array(bands_array, user_id)
    return jsonify(result), 200

@app.route("/outputs/<user_id>/<filename>")
def get_output(user_id, filename):
    user_out = os.path.join(OUTPUT_ROOT, str(user_id))
    return send_from_directory(user_out, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
    