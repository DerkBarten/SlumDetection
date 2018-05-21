from sklearn.manifold import Isomap
import numpy as np

from matplotlib import pyplot as plt
from util import read_geotiff
from groundtruth import create_mask, create_groundtruth
from analysis import get_feature_path, reshape_image

def create_dataset(image_path, blocks, scales, feature_names):
    shapefile = 'data/slums_approved.shp'
    image = np.array(read_geotiff(image_path))
    mask = create_mask(shapefile, image_path)
    groundtruth = create_groundtruth(mask, block_size=20, threshold=0)
    groundtruth = reshape_image(groundtruth, image.shape, 20, 150)

    
    features = None

    for feature_name in feature_names:
        for block in blocks:
            for scale in scales:
                
                f = get_feature(image_path, block, scale, feature_name)
                if features is None:
                    features = f
                else:
                    features = np.concatenate((features, f), axis=0)
    
    print(features.shape)
    features = np.reshape(features, (features.shape[1],  features.shape[2], features.shape[0]))
    print(features.shape)
    features[features == np.inf] = 0
    # plt.imshow(groundtruth > 0)
    # plt.show()
    # exit()
    true = features[groundtruth > 0]
    false = features[groundtruth < 1]
    print(true.shape)
    
    isomap = Isomap()
    res = isomap.fit_transform(true)
    plt.scatter(res[:,0], res[:,1])
    res = isomap.fit_transform(false)
    plt.scatter(res[:,0], res[:,1])
    plt.show()
    

def get_feature(image_path, block, scale, feature_name):
    feature_path = get_feature_path(image_path, block, scale, [1, 2, 3],
                                    feature_name)

    if feature_path is None:
        print("Error: cannot find feature file")
        return

    return np.array(read_geotiff(feature_path))

create_dataset('data/section_1.tif', [20], [150], ['hog', 'lsr'])