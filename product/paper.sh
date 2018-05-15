#!/bin/bash
source activate dynaslum2
set -euo pipefail

# Worldview2 image of Bangalore
image=~/projects/dynaslum/thesis/product/data/Bangalore.TIF
#image=~/projects/dynaslum/thesis/product/data/section_1_fixed.tif
output=~/projects/dynaslum/thesis/product/features

# GLCM PanTex (3)
#spfeas -i $image -o $output --block 8 --band-positions 5 3 2 --scales 8 16 32 --triggers pantex

# Histogram of Gradients (15)
spfeas -i $image -o $output --block 24 --band-positions 5 3 2 --scales 50 --triggers hog

# Lacunarity (3)
# Cannot compute at scales 15m (8 pixels) and 30m (16 pixels) because our data are 1.84m instead of 0.6m resolution,
# so the required r parameter would be smaller than the minimum meaningful value of 2.
#spfeas -i $image -o $output --block 8 --band-positions 5 3 2 --scales 32 --lac-r 2 --triggers lac

# Linear Feature Distribution (6)
# Not implemented

# Line Support Regions (9)
#spfeas -i $image -o $output --block 24 --band-positions 5 3 2 --scales 24 --triggers lsr

# Vegetation indices (2)
# Note that the command below also appears to compute the block standard deviation
#spfeas -i $image -o $output --block 8 --sensor WorldView2 --triggers pndvi rbvi

# SIFT (96)
# Not implemented

# TEXTONS (96)
# Not implemented (maybe partially implemented?)

# Stack features
#spfeas -i $image -o $output --block 8 --rgb --scales 8 16 32 64 --stack-only

# Classify:
# Input data: n_pixels * n_features
# Input labels: n_pixels * n_classes (2: slum/other)

source deactivate