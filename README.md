#RGB–NIR Multispectral Segmentation using K-Means + Elbow Method

This project performs unsupervised segmentation on RGB+NIR images (from the EPFL RGB-NIR multispectral dataset).
It automatically determines the optimal number of clusters (K) using the Elbow Method and then performs final segmentation using the best global K.

✅ Final Selected Global K = 4
Based on elbow analysis across all images, K=4 was chosen because it appeared most frequently as the best value for segmentation quality.
---------------------------------------------------------------------------------------------------------------------------------------------
Project Features
* Fuse RGB and NIR data for better vegetation/building/road separation
* Extract 7 spectral features from each pixel
* Automatically determine best K per image using the elbow method
* Compute global K statistics (K=4 was most common)
* Create color-coded segmentation maps
* Save segmentation outputs for all images
