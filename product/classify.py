import itertools
import os
import enum
import pickle
import numpy as np

from util import read_geotiff
from analysis import get_features
from groundtruth import create_mask, create_groundtruth, reshape_image
from groundtruth import overlay_groundtruth, create_dict

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN

import seaborn as sns
from tabulate import tabulate

imagefiles = {
    0: 'data/section_1.tif',
    1: 'data/section_2.tif',
    2: 'data/section_3.tif',
    3: 'data/section_4.tif',
    4: 'data/section_5.tif'
}
shapefile = 'data/slums_approved.shp'

classifiers = {
    # 1: SVC(),
    2: DecisionTreeClassifier(),
    3: RandomForestClassifier(),
    4: MLPClassifier(),
    5: AdaBoostClassifier(),
    6: GradientBoostingClassifier()
}


def load_features(block, scales, bands, feature_names, image_path):
    if os.path.exists(image_path):
        features = get_features(image_path, block, scales, bands,
                                feature_names)
        features[features == np.inf] = 0
        return features
    print("Feature does not exist")
    exit()


def create_dataset(path, block, scales, bands, feature):
    image = np.array(read_geotiff(path))
    mask = create_mask(shapefile, path)
    groundtruth = create_groundtruth(mask, block_size=block, threshold=0.6)
    groundtruth = reshape_image(groundtruth, (image.shape[1], image.shape[2]),
                                block, max(scales))

    #for f in feature:
        # tr = f[groundtruth > 0]
        # fa = f[groundtruth < 1]
        # print(tr.shape)
        # print(fa.shape)
        # sns.kdeplot(tr)
        # sns.kdeplot(fa)
        # plt.imshow(f)
        # plt.show()
        # break
    # exit()

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
    # print(X.shape)
    # print(y.shape)

    # X = np.reshape(X, feature.shape)
    # y = np.reshape(y, feature[0].shape)
    
    # for x in X:
    #     tr = x[y > 0]
    #     fa = x[y < 1]
        
    #     sns.kdeplot(tr)
    #     sns.kdeplot(fa)
    #     plt.show()
    # exit()

    # for i in range(10):
    #     x = X[:, i]
    #     tr = x[y > 0]
    #     fa = x[y < 1]
    #     sns.kdeplot(tr)
    #     sns.kdeplot(fa)
    #     plt.show()
    # exit()


    return X, y


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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def oversample(Xtrain, ytrain):
    # return RandomOverSampler().fit_sample(Xtrain, ytrain)
    return SMOTE().fit_sample(Xtrain, ytrain)
    #return ADASYN().fit_sample(Xtrain, ytrain)


def undersample(Xtrain, ytrain):
    return RandomUnderSampler(ratio='majority').fit_sample(Xtrain, ytrain)


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

    plt.imshow(np.reshape(clusters, shape[1:]))
    plt.show()


def classify(Xtrain, ytrain, Xtest, sklearn_classifier):
    sklearn_classifier.fit(Xtrain, ytrain)
    return sklearn_classifier.predict(Xtest)


def plot_dataset_distribution(Xtrain, Xtest):
    Xtrain = np.transpose(Xtrain)
    Xtest = np.transpose(Xtest)

    for i in range(Xtrain.shape[0]):
        train_feature = Xtrain[i]
        test_feature = Xtest[i]
        sns.kdeplot(train_feature)
        sns.kdeplot(test_feature)
        plt.show()


def create_train_test(train_images_paths, test_image_path, feature_names,
                      scales, block, bands):
    test_features = load_features(block, scales, bands, feature_names,
                                  test_image_path)
    Xtest, ytest = create_dataset(test_image_path, block, scales, bands,
                                  test_features)

    Xtrain = np.empty((0, Xtest.shape[1]))
    ytrain = np.ravel(np.empty((0, 1)))

    for path in train_images_paths:
        features = load_features(block, scales, bands, feature_names, path)
        X, y = create_dataset(path, block, scales, bands, features)

        Xtrain = np.concatenate((Xtrain, X), axis=0)
        ytrain = np.concatenate((ytrain, y), axis=0)

    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    Xtrain, ytrain = shuffle(Xtrain, ytrain)
    Xtrain, ytrain = oversample(Xtrain, ytrain)

    #Xtrain, ytrain = undersample(Xtrain, ytrain)
    return Xtrain, ytrain, Xtest, ytest, test_features.shape

def run_results():
    block = 20
    scales = [50, 100, 150]
    bands = [1, 2, 3]
    feature_name_list = [['hog', 'lsr', 'rid'], ['hog'], ['lsr'], ['rid']]
    images = [ [imagefiles[0], imagefiles[1], imagefiles[2]],
               [imagefiles[1], imagefiles[2], imagefiles[0]],
               [imagefiles[2], imagefiles[0], imagefiles[1]] ]

    for traintest in images:
        test_image = traintest[0]
        train_images = traintest[1:]
        for feature_names in feature_name_list:
            Xtrain, ytrain, Xtest, ytest, shape =\
                create_train_test(train_images, test_image, feature_names,
                                  scales, block, bands)
            print("Table: {} {} {} {}".format(feature_names, scales,
                                              test_image, train_images))
            table = np.empty((0, 4))
            for i in classifiers:
                prediction = classify(Xtrain, ytrain, Xtest, classifiers[i])
                precison = metrics.precision_score(ytest, prediction)
                f1_score = metrics.f1_score(ytest, prediction)
                keepers = np.where(np.logical_not((np.vstack((ytest,
                                   prediction)) == 0).all(axis=0)))
                jaccard = metrics.jaccard_similarity_score(ytest[keepers],
                                                           prediction[keepers])
                name = str(classifiers[i]).split("(")[0]
                entry = np.array([[name, precison, f1_score, jaccard]])
                table = np.concatenate((table, entry))
            
            # 50% slum
            prediction = np.random.randint(2, size=ytest.shape[0])
            precison = metrics.precision_score(ytest, prediction)
            f1_score = metrics.f1_score(ytest, prediction)
            keepers = np.where(np.logical_not((np.vstack((ytest,
                               prediction)) == 0).all(axis=0)))
            jaccard = metrics.jaccard_similarity_score(ytest[keepers],
                                                       prediction[keepers])
            entry = np.array([["50% Random Noise as slum", precison, f1_score,
                               jaccard]])
            table = np.concatenate((table, entry))

            # 100% slum
            prediction = np.ones(ytest.shape[0])
            precison = metrics.precision_score(ytest, prediction)
            f1_score = metrics.f1_score(ytest, prediction)
            keepers = np.where(np.logical_not((np.vstack((ytest,
                               prediction)) == 0).all(axis=0)))
            jaccard = metrics.jaccard_similarity_score(ytest[keepers],
                                                       prediction[keepers])
            entry = np.array([["100% as slum", precison, f1_score,
                               jaccard]])
            table = np.concatenate((table, entry))

            # Max performance
            entry = np.array([["Max", np.max(table[:, 1].astype(float)),
                               np.max(table[:, 2].astype(float)),
                               np.max(table[:, 3].astype(float))]])

            table = np.concatenate((table, entry))

            print(tabulate(table, headers=["Classifier", "Precision",
                           "F1-Score", "Jaccard's Index"], tablefmt='orgtbl'))
            print("")


def test():
    block = 20
    scales = [50, 100, 150]
    bands = [1, 2, 3]
    feature_names = ['lsr', 'hog', 'rid']
    classifier = classifiers[3]
    train_images = [imagefiles[2], imagefiles[3]]
    test_image = imagefiles[0]


    Xtrain, ytrain, Xtest, ytest, shape =\
                create_train_test(train_images, test_image, feature_names,
                                  scales, block, bands)

    

    # for i in range(10):
    #     print(i)
    #     f = Xtrain[:, i]
    #     sns.kdeplot(f[ytrain > 0])
    #     sns.kdeplot(f[ytrain < 1])
    #     plt.show()
    # exit()
    prediction = classify(Xtrain, ytrain, Xtest, classifier)

    plot_confusion_matrix(confusion_matrix(ytest, prediction),
                          classes=['Non Slum', 'Slum'],
                          title='Confusion matrix')
    plt.show()

    image = np.array(read_geotiff(test_image))
    image = np.dstack((image[0], image[1], image[2]))
    overlay_groundtruth(np.reshape(prediction, shape[1:]), image, block)


if __name__ == "__main__":
    # run_results()
    test()
   
