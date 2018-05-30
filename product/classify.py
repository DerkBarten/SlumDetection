from sklearn.manifold import Isomap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans

import numpy as np

from matplotlib import pyplot as plt
from util import read_geotiff
from groundtruth import create_mask, create_groundtruth, reshape_image
from analysis import get_features
from sklearn.svm import SVC

from sklearn.manifold import TSNE
from sklearn.utils import shuffle

import itertools
import os
import enum
import pickle

basedir = "data"
imagefiles = { 0 : 'section_1.tif',
               1 : 'section_2.tif',
               2 : 'section_3.tif'}


class ClassifierType:
    FORREST, KNN, SVM = range(3)


def get_classifier(class_type):
    if class_type == ClassifierType.FORREST:
        return RandomForestClassifier(max_depth=6)
    if class_type == ClassifierType.KNN:
        return KNeighborsClassifier()
    if class_type == ClassifierType.SVM:
        return SVC()
    print("Error: Invalid Classifier type")
    exit()


def load_features(block, scales, bands, feature_names):
    features = {}
    for index in imagefiles:
        path = os.path.join(basedir, imagefiles[index])
        if os.path.exists(path):
            features[index] = get_features(path, block, scales, bands,
                                           feature_names)
            features[index][features[index] == np.inf] = 0
        else:
            print("Feature does not exist")
    return features


def create_dataset(path, block, scales, bands, feature):
    shapefile = 'data/slums_approved.shp'
    
    image = np.array(read_geotiff(path))
    mask = create_mask(shapefile, path)
    groundtruth = create_groundtruth(mask, block_size=block, threshold=0.5)
    image_shape = (image.shape[1], image.shape[2])
    groundtruth = reshape_image(groundtruth, image_shape, block, max(scales))

    # shape = feature.shape
    # print(shape)
    # plt.imshow(feature[0])
    # plt.show()
    X = np.reshape(feature, (feature.shape[1] * feature.shape[2],
                             feature.shape[0]))
    
    for i, feature in enumerate(X):
        X[i] = preprocessing.scale(feature)

    # image = image[0:-1]

    # plt.imshow(image[0])
    # plt.show()
    # plt.imshow(np.array(groundtruth > 0, dtype=int))
    # plt.show()
   
    # print(groundtruth.shape)
    # plt.imshow(np.array(groundtruth > 0, dtype=int))
    # plt.show()

    y = np.ravel(np.array(groundtruth > 0, dtype=int))

    # print(X.shape)
    
    # print(y.shape)
    # print(np.bincount(y))
    return X, y

def create_train_test(d0, d1, d2):
    Xtest, ytest = d0

    X1, y1 = d1
    X2, y2 = d2

    Xtrain = np.concatenate((X1, X2), axis=0)
    ytrain = np.concatenate((y1, y2), axis=0)
    Xtrain, ytrain = shuffle(Xtrain, ytrain)

    return Xtrain, ytrain, Xtest, ytest


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

def classify_Gaussian(Xtrain, ytrain, Xtest, ytest):
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xtest)
    cnf_matrix = confusion_matrix(ytest, ypred)
    plot_confusion_matrix(cnf_matrix, classes=['formal', 'informal'],
                          title='Confusion matrix')
    plt.show()
    return ypred


def classify_Gradient(Xtrain, ytrain, Xtest, ytest):
    from sklearn.ensemble import GradientBoostingClassifier
    grad = GradientBoostingClassifier()
    grad.fit(Xtrain, ytrain)

    ypred = grad.predict(Xtest)
    cnf_matrix = confusion_matrix(ytest, ypred)
    plot_confusion_matrix(cnf_matrix, classes=['formal', 'informal'],
                          title='Confusion matrix')
    plt.show()
    return ypred


def oversample(Xtrain, ytrain):
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler()
    return ros.fit_sample(Xtrain, ytrain)


def do_tsne(Xtrain, shape, load=False, save=True):
    print("performing tsne")

    if load:
        f = open('cluster.pkl', 'r')
        points = pickle.load(f)
    else:
        tsne = TSNE(n_components=2, init='random', random_state=0)
        points = tsne.fit_transform(Xtrain)

    if save:
        f = open('cluster.pkl', 'w')
        pickle.dump(points, f)

    print(points.shape)

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(points)
    print(clusters.shape)

    plt.imshow(np.reshape(clusters, shape))
    plt.show()

def classify(Xtrain, ytrain, Xtest, ytest, class_type):
    classifier = get_classifier(class_type)
    classifier.fit(Xtrain, ytrain)
    return classifier.predict(Xtest)

block = 20
scales = [150]
bands = [1, 2, 3]
features = ['lsr', 'hog', 'rid']

features = load_features(block, scales, bands, features)

d0 = create_dataset(os.path.join(basedir, imagefiles[0]), block, scales, bands,
                    features[0])
# d1 = create_dataset(os.path.join(basedir, imagefiles[1]), block, scales, bands,
#                     features[1])
# d2 = create_dataset(os.path.join(basedir, imagefiles[2]), block, scales, bands,
#                     features[2])

Xtrain, _ = d0
print(features[0].shape[1:])
do_tsne(Xtrain, features[0].shape[1:])

# shape = features[0].shape[1:]

# Xtrain, ytrain, Xtest, ytest = create_train_test(d0, d1, d2)

# Xtrain, ytrain = oversample(Xtrain, ytrain)

# prediction = classify(Xtrain, ytrain, Xtest, ytest, ClassifierType.FORREST)

# plot_confusion_matrix(confusion_matrix(ytest, prediction),
#                       classes=['formal', 'informal'],
#                       title='Confusion matrix')
# plt.show()

# print(shape)
# print(prediction.shape)
# plt.imshow(np.reshape(prediction, shape))
# plt.show()
