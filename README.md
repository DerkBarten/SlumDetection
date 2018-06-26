## Description
This repository contains all code, data and results from the thesis:*The Detection of Morphological Slums in
Bangalore from Satellite Image*.

## Usage
The general workflow for slum detection consists of two steps,  the creation and classification of features.

1. The feature creation  is handled in `feature.py`,  this module contains the `Feature` class creates the features as specified in it's parameters. An example of the feature calculation is located in the bottom of the `analysis.py` script, which automatically creates and analyzes the features. All features are saved to `product/feature`,  the plots created by `analysis.py` are stored in `product/analysis`.

2. The  classification is performed using the `classify.py` script. This script contains the `Dataset` class, which creates a dataset from the calculated features in the previous section. It requires the same parameters used which were used for the created features, as it will return an error if it cannot find the files specified by its parameters. This dataset is used by the `Classify` class to perform the classification task, which stores the  classification results in `product/results`. An example of the usage of the script can be found at the bottom of `classify.py`.

## Data and Features
The data we have used in the thesis is available at https://derkbarten.nl/thesis. Here you can also download all results, features and feature analysis plots. Regarding the data, the example code used three sections and shape files that are included in `data.zip`. The contents of the zip file should be placed in the product folder, creating the path to, for example, section 1 as `product/data/section_1.tif`.

Regarding the features, if you would like to use the precomputed features, these are stored in `computed_features.zip` on the same webpage. Like the data folder, these features have to be placed in the product folder in the same way. I recommend to compute the features yourself as some features might not be compatible anymore to the current codebase.   


## Folder naming convention
The features stored in `product/features` are stored using a specific naming convention to be able to automatically load feature files using parameters such as scale and block size as opposed to explicitly passing on a file path. For instance, the folder `section_1__BD1-2-3_BK20_SC50-100-150_TRhog`is created from the first section;  the image bands 1, 2, and 3; a block size of 20;  the scales 50, 100 and150; the Histogram of Oriented Gradients Feature. All scripts that handle feature IO use this notation to store and load features. Furthermore, the files created by `analysis.py`stored in `product/analysis` and the results in `product/results`use a variation of the same notation. 

## Dependencies:
The dependencies are listed in the `requirements.txt`. If you only intend to use RID, you can omit the last requirement, which is the library that is used for HoG and LSR. 
