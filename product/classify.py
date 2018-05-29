from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import numpy as np

from matplotlib import pyplot as plt
from util import read_geotiff
from groundtruth import create_mask, create_groundtruth, reshape_image
from analysis import get_features

from sklearn.manifold import TSNE

import itertools


def create_dataset(image_path, block, scale, bands, feature_names):
    shapefile = 'data/slums_approved.shp'
    image = np.array(read_geotiff(image_path))
    mask = create_mask(shapefile, image_path)
    groundtruth = create_groundtruth(mask, block_size=block, threshold=0.5)
    image_shape = (image.shape[1], image.shape[2])
    groundtruth = reshape_image(groundtruth, image_shape, block, scale)

    features = None
    for feature_name in feature_names:
        f = get_features(image_path, block, scale, bands, feature_name)
        if features is None:
            features = f
        else:
            features = np.concatenate((features, f), axis=0)

    features = np.reshape(features, (features.shape[1],  features.shape[2], features.shape[0]))
    features[features == np.inf] = 0

    true = features[groundtruth > 0]
    false = features[groundtruth < 1]

    X = np.concatenate((false, true), axis=0)
    y = np.ravel(np.concatenate((np.zeros((false.shape[0], 1)),
                                 np.ones((true.shape[0], 1)))))
    return (X, y)


def balance_dataset(X, y, class_ratio=1.3):
    """
    Balances the dataset over the classes
    Class ratio 1.5 means that there are 50% more non-slum examples compared to slum examples

    Zero and one reference the class
    filters part of the X matrix out, where y = 0
    :param X:
    :param y:
    :param class_ratio:
    """
    to_take = round(len(y[y == 1]) * class_ratio)

    X_zeros = X[y == 0, :]
    y_zeros = y[y == 0]

    row_indices = np.random.choice(X_zeros.shape[0], int(to_take), replace=False)
    X_zeros = X_zeros[row_indices, :]
    y_zeros = y_zeros[row_indices]

    X_ones = X[y == 1, :]
    y_ones = y[y == 1]

    X = np.append(X_zeros, X_ones, axis=0)
    y = np.append(y_zeros, y_ones)

    return X, y

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def classify_Forrest():
    X, y = create_dataset('data/section_1.tif', 20, 100, [1, 2, 3], ['lsr', 'rid'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = RandomForestClassifier(max_depth=4)
    X_train, y_train = balance_dataset(X_train, y_train)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=['formal', 'informal'],
                          title='Confusion matrix')
    plt.show()


def classify_TSNE():
    X, y = create_dataset('data/section_1.tif', 20, 100, [1,2,3], ['lsr', 'rid'])
    # X = np.matrix(X)
    # X = X[:, [1,4]]
    
    output = TSNE(n_components=2).fit_transform(X)
    print(output.shape)
    print(X.shape)
    print(y.shape)
    X0 = np.ravel(output[:, 0])
    X1 = np.ravel(output[:, 1])
    plt.scatter(X0[y == 0], X1[y == 0], c='r')
    plt.scatter(X0[y == 1], X1[y == 1], c='b')
    plt.show()

classify_TSNE()
#classify_Forrest()