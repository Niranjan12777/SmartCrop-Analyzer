import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image
from tensorflow.keras.models import load_model
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "unet_crop_analysis.h5"
INPUT_TIFF = "sentinel2_model_ready.tiff"
PATCH_SIZE = 32
STRIDE = PATCH_SIZE // 3

# ------------------------------
# CLASS LABELS (Indian Pines Dataset)
# ------------------------------
class_labels = [
    "Background", "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"
]

# Moisture colormap
moisture_cmap = LinearSegmentedColormap.from_list("moisture_cmap", ["red","orange", "white", "blue"])

# ------------------------------
# LOAD MODEL
# ------------------------------
print("Loading trained model...")
model = load_model(MODEL_PATH)

# ------------------------------
# LOAD TIFF IMAGE
# ------------------------------
print(f"Loading image: {INPUT_TIFF}")
with rasterio.open(INPUT_TIFF) as src:
    img = src.read()  # shape = (bands, H, W)
    img = reshape_as_image(img)  # (H, W, Bands)

print(f"Input image shape: {img.shape}")

# Normalize input
img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
H, W, C = img_norm.shape

pred_probs = np.zeros((H, W, len(class_labels)), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)

print("Running model prediction with smoothing...")
for i in range(0, H - PATCH_SIZE + 1, STRIDE):
    for j in range(0, W - PATCH_SIZE + 1, STRIDE):
        patch = img_norm[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
        patch = np.expand_dims(patch, axis=0)

        pred = model.predict(patch, verbose=0)[0]  # (PATCH, PATCH, num_classes)

        pred_probs[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += pred
        count_map[i:i+PATCH_SIZE, j:j+PATCH_SIZE] += 1

# Avoid division by zero
count_map[count_map == 0] = 1
pred_probs /= count_map[..., np.newaxis]

# Final class map + health/stress
health_map = np.max(pred_probs, axis=-1)         # confidence = health
stress_map_conf = 1 - health_map                 # confidence-based stress

print("✅ Crop classification + health mapping complete.")

# ------------------------------
# STRESS INDEX (NDRE)
# ------------------------------

nir = img[:, :, 7].astype(np.float32)
red_edge = img[:, :, 4].astype(np.float32)

ndre = (nir - red_edge) / (nir + red_edge + 1e-6)
ndre_norm = (ndre - np.min(ndre)) / (np.max(ndre) - np.min(ndre))  # 0–1
stress_map_ndre = 1 - ndre_norm  # invert: high value = more stress

# ------------------------------
# MOISTURE INDEX (NDMI)
# ------------------------------

nir_narrow = img[:, :, 8].astype(np.float32)   # B8A
swir1 = img[:, :, 10].astype(np.float32)       # B11

ndmi = (nir_narrow - swir1) / (nir_narrow + swir1 + 1e-6)  # avoid div by zero
ndmi_norm = (ndmi - np.min(ndmi)) / (np.max(ndmi) - np.min(ndmi))  # scale 0–1


# ------------------------------
# VISUALIZATION 1: Crop Health (confidence-based, green → red)
# ------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(health_map, cmap="RdYlGn")
plt.colorbar(label="Crop Health (Green=Good, Red=Poor)")
plt.title("Crop Health Map")
plt.axis("off")
plt.show()

# ------------------------------
# VISUALIZATION 2: Crop Stress (NDRE-based)
# ------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(stress_map_ndre, cmap="Reds")
plt.colorbar(label="Crop Stress (Red=High)")
plt.title("Crop Stress Map")
plt.axis("off")
plt.show()

# ------------------------------
# VISUALIZATION 3: Moisture Index (NDMI, red → blue)
# ------------------------------
plt.figure(figsize=(8, 6))
plt.imshow(ndmi_norm, cmap=moisture_cmap)
plt.colorbar(label="Moisture Index (Red=Dry, Blue=Moist)")
plt.title("Crop Moisture Index")
plt.axis("off")
plt.show()

