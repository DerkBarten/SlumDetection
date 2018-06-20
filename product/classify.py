import itertools
import os
import enum
import pickle
import logging
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from util import read_geotiff, Image
from feature import Feature
from groundtruth import create_mask, create_groundtruth, reshape_image

from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import shift

LOG = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

shapefile = 'data/slums_approved.shp'


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Little more room for the X label
    plt.gcf().subplots_adjust(bottom=0.2)


def get_metrics(prediction, ytest):
    f1_score = metrics.f1_score(ytest, prediction)
    matthews = metrics.matthews_corrcoef(ytest, prediction)
    return f1_score, matthews


class Dataset:
    def __init__(self, train_images, test_image, shapefile,
                 feature_names, scales=[50, 100, 150], block_size=20,
                 bands=[1, 2, 3]):
        self.train_images = train_images
        self.test_image = test_image
        self.shapefile = shapefile
        self.feature_names = feature_names
        self.bands = bands
        self.scales = scales
        self.block_size = block_size
        self.dataset = None
        self.feature_shape = None
        self._create_train_test()

    def _load_features(self, image):
        if os.path.exists(image.path):
            features = Feature(image, self.block_size, self.scales,
                               self.bands, self.feature_names).get()
            
            features[features == np.inf] = 0
            return features
        print("Feature does not exist")
        exit()

    def _create_dataset(self, image, feature):
        mask = create_mask(shapefile, image.path)
        groundtruth = create_groundtruth(mask, block_size=self.block_size,
                                         threshold=0.6)
        groundtruth = reshape_image(groundtruth, image.shape, self.block_size,
                                    max(self.scales))

        X = []
        for i in range(feature.shape[1]):
            for j in range(feature.shape[2]):
                X.append(feature[:, i, j])

        y = []
        for i in range(groundtruth.shape[0]):
            for j in range(groundtruth.shape[1]):
                y.append(groundtruth[i, j])

        X = np.array(X)
        y = np.array(y)
        return X, y

    def _create_train_test(self):
        test_features = self._load_features(self.test_image)
        Xtest, ytest = self._create_dataset(self.test_image, test_features)

        Xtrain = np.empty((0, Xtest.shape[1]))
        ytrain = np.ravel(np.empty((0, 1)))

        for image in self.train_images:
            train_features = self._load_features(image)
            X, y = self._create_dataset(image, train_features)
            # print(X.shape)
            # print(y.shape)

            Xtrain = np.concatenate((Xtrain, X), axis=0)
            ytrain = np.concatenate((ytrain, y), axis=0)

        # print(Xtrain.shape)
        # print(ytrain.shape)

        scaler = StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)


        Xtrain, ytrain = shuffle(Xtrain, ytrain)
        Xtrain, ytrain = SMOTE().fit_sample(Xtrain, ytrain)

        self.dataset = (Xtrain, ytrain, Xtest, ytest)
        self.feature_shape = test_features[0].shape

    def get_dataset(self):
        return self.dataset

    def get_feature_shape(self):
        return self.feature_shape

    def get_train_images(self):
        return self.train_images

    def get_test_image(self):
        return self.test_image

    def get_feature_names(self):
        return self.feature_names

    def get_scales(self):
        return self.scales

    def get_block_size(self):
        return self.block_size


class Classify:
    classifiers = {
        0: DecisionTreeClassifier(),
        1: RandomForestClassifier(),
        2: MLPClassifier(),
        3: AdaBoostClassifier(),
        4: GradientBoostingClassifier()
    }

    def __init__(self, dataset, classifier_indices=None, experiments=5):
        self.dataset = dataset
        self.classifier_indices = classifier_indices
        if not self.classifier_indices:
            self.classifier_indices = list(self.classifiers.keys())
        self.experiments = experiments

        self.train_images = dataset.get_train_images()
        self.test_image = dataset.get_test_image()
        self.feature_names = dataset.get_feature_names()
        self.scales = dataset.get_scales()
        self.block_size = dataset.get_block_size()

    def _get_classifier_name(self, index):
        return str(self.classifiers[index]).split("(")[0]

    def _create_confusion(self, prediction, ytest, folder, classifier_index):
        basename = self._get_basename()
        classifier_name = self._get_classifier_name(classifier_index)
        name = "confusion_" + basename + "_" + classifier_name + ".png"
        path = os.path.join(folder, name)
        matrix = metrics.confusion_matrix(ytest, prediction)
        plot_confusion_matrix(matrix,
                              classes=['Formal', 'Informal'],
                              title='Confusion Matrix')
        LOG.info("Saving confusion as: {}".format(path))
        plt.savefig(path, format='png', dpi=200)
        plt.clf()

    def _create_overlay(self, prediction, ytest, folder, classifier_index):
        basename = self._get_basename()
        classifier_name = self._get_classifier_name(classifier_index)
        name = "overlay_" + basename + "_" + classifier_name + ".png"
        path = os.path.join(folder, name)

        prediction = np.reshape(prediction, self.dataset.get_feature_shape())
        prediction = shift(prediction, 3, cval=0)
        prediction = zoom(prediction, self.block_size, order=0)
        # Compensate for the padding
        plt.axis('off')
        image = self.test_image.RGB
        #image = np.dstack((image[0], image[1], image[2]))
        plt.imshow(image)
        plt.imshow(prediction, alpha=0.5)
        LOG.info("Saving overlay as: {}".format(path))
        plt.savefig(path, format='png', dpi=400)
        plt.clf()

    def _create_metrics(self, folder):
        columns = ["F1 score", "Matthews"]
        indices = [self._get_classifier_name(index)
                   for index in self.classifier_indices]
        metrics = pd.DataFrame(columns=columns)

        LOG.info("Start creating metrics")
        for index in self.classifier_indices:
            tmp = np.empty((0, 2))
            classifier = self.classifiers[index]
            Xtrain, ytrain, Xtest, ytest = self.dataset.get_dataset()
            for i in range(self.experiments):
                LOG.info("Experiment {}/{}".format(index * i + i, self.experiments * len(self.classifier_indices)))
                classifier.fit(Xtrain, ytrain)
                prediction = classifier.predict(Xtest)

                f1_score, matthews = get_metrics(prediction, ytest)

                entry = np.array([[f1_score, matthews]])
                tmp = np.concatenate((tmp, entry))

            mean = np.mean(tmp, axis=0)
            metrics = metrics.append(pd.DataFrame(np.reshape(mean, (1, 2)),
                                     index=[self._get_classifier_name(index)],
                                     columns=columns))

        basename = self._get_basename()
        name = "metrics_" + basename + "_" + ".csv"
        path = os.path.join(folder, name)
        LOG.info("Saving metrics as: {}".format(path))
        metrics.to_csv(path, sep=',')

    def _get_basename(self):
        tr_string = ""
        for image in self.train_images:
            tr_string += "_" + image.filename
        te_string = self.test_image.filename

        sc_string = ""
        for scale in self.scales:
            sc_string += "_" + str(scale)

        f_string = ""
        for feature in self.feature_names:
            f_string += "_" + feature

        return "TR{}_TE_{}_BK_{}_SC{}_F{}".format(tr_string, te_string,
                                                  self.block_size,
                                                  sc_string,  f_string)

    def classify(self):
        folder = os.path.join("results", self._get_basename())
        if not os.path.exists(folder):
            os.mkdir(folder)

        Xtrain, ytrain, Xtest, ytest = self.dataset.get_dataset()

        LOG.info("Classifying {}".format(self._get_basename()))
        for index in self.classifier_indices:
            LOG.info("Classifying using {}".format(self._get_classifier_name(index)))
            classifier = self.classifiers[index]
            classifier.fit(Xtrain, ytrain)
            prediction = classifier.predict(Xtest)

            self._create_confusion(prediction, ytest, folder, index)
            self._create_overlay(prediction, ytest, folder, index)
        self._create_metrics(folder)

if __name__ == "__main__":
    scales_list = [[50, 100, 150]]
    block_size_list = [20]
    features = ['hog']

    section_1 = Image('data/section_1.tif')
    section_2 = Image('data/section_2.tif')
    section_3 = Image('data/section_3.tif')

    test_image = section_2
    train_images = [section_1, section_3]

    for block_size in block_size_list:
        for scales in scales_list:
            dataset = Dataset(train_images, test_image,
                              shapefile, features, scales=scales,
                              block_size=block_size)
            classify = Classify(dataset)
            classify.classify()
