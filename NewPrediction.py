import os
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image
from tensorflow.keras.models import load_model
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_PATH = "Browser_images(23)"      # Can be a folder OR a ZIP file
MODEL_PATH = "unet_crop_analysis.h5"  # Your trained U-Net model
OUTPUT_DIR = "Outputs"                # Output folder for maps
PATCH_SIZE = 32
STRIDE = PATCH_SIZE // 3
EXPECTED_BANDS = 13                   # Expected number of input bands

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FUNCTION 1 — LOAD BAND FILES FROM FOLDER OR ZIP
# ============================================================

def load_bands_from_input(input_path):
    """
    Loads Sentinel-2 bands from FOLDER or ZIP and returns stacked array.
    """

    # If ZIP → Extract to temp folder
    if input_path.lower().endswith(".zip"):
        print("ZIP file detected. Extracting...")
        extract_dir = "temp_extracted_bands"
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        folder = extract_dir

    else:
        folder = input_path

    # Collect .tif files
    band_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff"))
    ])

    if not band_files:
        raise FileNotFoundError("No .tif files found in input.")

    print("Bands found:", band_files)

    # Read all bands
    bands = []
    meta = None
    for f in band_files:
        fp = os.path.join(folder, f)
        with rasterio.open(fp) as src:
            band = src.read(1)
            bands.append(band)
            if meta is None:
                meta = src.meta.copy()

    # Stack
    bands_array = np.stack(bands, axis=0)
    available_bands = bands_array.shape[0]

    # Pad or trim
    if available_bands < EXPECTED_BANDS:
        pad_count = EXPECTED_BANDS - available_bands
        pad = np.zeros((pad_count, bands_array.shape[1], bands_array.shape[2]))
        bands_array = np.concatenate([bands_array, pad], axis=0)
        print(f"Padded missing {pad_count} bands.")
    elif available_bands > EXPECTED_BANDS:
        bands_array = bands_array[:EXPECTED_BANDS]
        print(f"Trimmed to {EXPECTED_BANDS} bands.")

    return bands_array


# ============================================================
# FUNCTION 2 — RUN U-NET PREDICTION & SAVE MAPS
# ============================================================

def generate_maps_from_array(bands_array):

    print("\nPreparing input data for prediction...")
    img = reshape_as_image(bands_array)    # (H, W, C)
    img_norm = (img - img.min()) / (img.max() - img.min())

    H, W, C = img_norm.shape

    # Load model
    print("Loading U-Net model...")
    model = load_model(MODEL_PATH)

    pred_probs = np.zeros((H, W, 17), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    print("Running patch-based prediction...")
    for i in range(0, H - PATCH_SIZE + 1, STRIDE):
        for j in range(0, W - PATCH_SIZE + 1, STRIDE):

            patch = img_norm[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
            patch = np.expand_dims(patch, 0)

            pred = model.predict(patch, verbose=0)[0]

            pred_probs[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += pred
            count_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += 1

    count_map[count_map == 0] = 1
    pred_probs /= count_map[..., None]

    print("Pixel-wise prediction complete.")

    # -------------------------------------
    # HEALTH MAP
    # -------------------------------------
    health_map = np.max(pred_probs, axis=-1)

    # -------------------------------------
    # NDRE STRESS MAP
    # -------------------------------------
    nir = img[:, :, 7]
    red_edge = img[:, :, 4]
    ndre = (nir - red_edge) / (nir + red_edge + 1e-6)
    ndre_norm = (ndre - ndre.min()) / (ndre.max() - ndre.min())
    stress_map = 1 - ndre_norm

    # -------------------------------------
    # NDMI MOISTURE MAP
    # -------------------------------------
    nir_narrow = img[:, :, 8]
    swir1 = img[:, :, 10]

    ndmi = (nir_narrow - swir1) / (nir_narrow + swir1 + 1e-6)
    ndmi_norm = (ndmi - ndmi.min()) / (ndmi.max() - ndmi.min())

    moisture_cmap = LinearSegmentedColormap.from_list("moisture", ["red", "orange", "white", "blue"])

    # Save helper
    def save_map(image, cmap, title, filename):
        path = os.path.join(OUTPUT_DIR, filename)
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap=cmap)
        plt.colorbar(label=title)
        plt.title(title)
        plt.axis("off")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("📁 Saved:", path)

    # Save all maps
    save_map(health_map, "RdYlGn", "Crop Health Map", "health_map.png")
    save_map(stress_map, "Reds", "Crop Stress Map (NDRE)", "stress_map.png")
    save_map(ndmi_norm, moisture_cmap, "Crop Moisture Map (NDMI)", "moisture_map.png")

    print("\nAll maps generated and saved in Outputs/ folder!")


# ============================================================
#                     MAIN PIPELINE
# ============================================================

print("==============================================")
print("CROP ANALYSIS — FULL PIPELINE STARTED")
print("==============================================")

bands_array = load_bands_from_input(INPUT_PATH)
generate_maps_from_array(bands_array)

print("\nPipeline completed successfully!\n")
