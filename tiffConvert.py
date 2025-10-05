import rasterio
import os
import numpy as np

# -------------------------------
# USER SETTINGS
# -------------------------------
input_folder = "Browser_images"  # Path to folder with Sentinel-2 bands
output_file = "sentinel2_model_ready.tiff"    # Output TIFF
expected_bands = 13  # Number of bands your model expects (hyperspectral model)
# -------------------------------

# Get all .tif or .tiff files
all_tif_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".tif", ".tiff"))
]

if not all_tif_files:
    raise FileNotFoundError(f"No .tif or .tiff files found in {input_folder}")

# Sort alphabetically to ensure band order
band_files = sorted(all_tif_files, key=lambda x: x.lower())
print("Bands found:", band_files)

# Read all available bands
bands = []
meta = None
for f in band_files:
    filepath = os.path.join(input_folder, f)
    with rasterio.open(filepath) as src:
        bands.append(src.read(1))
        if meta is None:
            meta = src.meta.copy()

# Safety check
if meta is None:
    raise RuntimeError("Could not read metadata from any file.")

# Convert to NumPy array
bands_array = np.stack(bands, axis=0)  # shape: (available_bands, H, W)
available_bands = bands_array.shape[0]

# -------------------------------
# PAD OR TRIM BANDS
# -------------------------------
if available_bands < expected_bands:
    pad_count = expected_bands - available_bands
    pad_array = np.zeros((pad_count, bands_array.shape[1], bands_array.shape[2]), dtype=bands_array.dtype)
    bands_array = np.concatenate([bands_array, pad_array], axis=0)
    print(f"📦 Padded with {pad_count} zero bands to match {expected_bands} expected bands.")
elif available_bands > expected_bands:
    bands_array = bands_array[:expected_bands, :, :]
    print(f"✂ Trimmed to {expected_bands} bands to match model input.")

# -------------------------------
# Save combined multi-band TIFF
# -------------------------------
meta.update(count=expected_bands)

with rasterio.open(output_file, "w", **meta) as dst:
    for idx in range(expected_bands):
        dst.write(bands_array[idx], idx + 1)

print(f"✅ Model-ready TIFF saved: {output_file}")