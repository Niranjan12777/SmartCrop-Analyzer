# SmartCrop-Analyzer
SmartCrop-Analyzer is an satellite-based agricultural monitoring system that leverages deep learning and spectral analysis to assess crop health, detect stress, and map moisture conditions. Using hyperspectral and multispectral imagery, the model applies UNet/CNN architectures along with vegetation indices to generate fine-resolution maps.

#Dataset Normalization: 
Before training, hyperspectral data undergoes preprocessing to remove:
Noisy spectral bands, Water absorption bands, and other Atmospheric bands by running the Normalize.py script.

# Model Training (UNet-based Training):
The patches are already uploaded after running the Normalize.py, to train the model we have to use the patches which are unet_label_masks_s2.npy and unet_patches_s2.npy and also install the required libraries before training the model. After the model is we get unet_crop_analysis.h5 model which we can use it further to predict and Map the plots.

# Prediction and Mapping:
This loads the trained Unet model and uses it to predict on the crop health maps, and other maps such as crop stress map and crop moisture index by indexing.
For the input use the browser_images folder, as it contains the georeferenced images which are used from the sentinel-2 data.
This will generate all the 3 Maps, each output displayed can be exported as a .png

