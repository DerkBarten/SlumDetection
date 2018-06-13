import os
import glob
import logging
import numpy as np

from util import read_geotiff
from rid import  Kernel, RoadIntersections, RoadIntersectionDensity, ktype

LOG = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


class Feature:
    """
    Args:
        image_path:     The name of the image the feature file was created
                        from; string.
        block:          The block size used in the creation of the feature;
                        integer.
        scale:          The scale used in the creation of the feature, integer.
        bands:          The color bands used in the creation of the feature;
                        array of integers.
        feature_name:   The name of the feature, this is either 'hog' or
                        'lsr'; string.
    """
    @classmethod
    def __init__(self, image_path, block_size=20, scales=[50], bands=[1, 2, 3],
                 feature_names=['hog']):
        self._image_path = image_path
        self._block_size = block_size
        self._scales = scales
        self._bands = bands
        self._feature_names = feature_names
        self._feature = None

    @classmethod
    def __get_folder_name(self, feature_names=None, scales=None):
        """
        Finds the path to the feature file based on the parameters used in its
        creation.

        Returns:
            The path to the feature file if it exists, else it returns None.

        """
        if not feature_names:
            feature_names = self._feature_names
        if not scales:
            scales = self._scales

        scale_string = Feature.get_scale_string(self._scales)
        feature_string = Feature.get_feature_string(sorted(feature_names),
                                                    'TR')

        folder = "features/features/{}__BD{}-{}-{}_BK{}_{}_{}/".format(
                    os.path.basename(os.path.splitext(self._image_path)[0]),
                    self._bands[0], self._bands[1], self._bands[2],
                    self._block_size, scale_string, feature_string)
        return folder

    def __get_feature_from_folder(self, folder):
        path = os.path.join(folder, '*')
        path = glob.glob(path)

        if path and os.path.exists(path[0]):
            return path[0]
        return None

    @staticmethod
    def get_scale_string(scales):
        scale_string = 'SC'
        for i, scale in enumerate(scales):
            if i == len(scales) - 1:
                scale_string += str(scale)
            else:
                scale_string += '{}-'.format(scale)
        return scale_string

    @staticmethod
    def get_feature_string(names, prefix=''):
        feature_string = prefix
        for i, name in enumerate(names):
            if i == len(names) - 1:
                feature_string += str(name)
            else:
                feature_string += '{}-'.format(name)
        return feature_string

    def create(self):
        """
        This function creates feature files by calling the spfeas program
        externally.

        """
        spfeas_features = sorted(set(self._feature_names).
                                 intersection(['hog', 'lsr']))
        rid_feature = sorted(set(self._feature_names).intersection(['rid']))

        if spfeas_features:
            LOG.info("Creating Spfeas feature {}".format(spfeas_features))
            folder = self.__get_folder_name(feature_names=spfeas_features)

            if self.__get_feature_from_folder(folder):
                LOG.info("Feature already exists, skipping ...")
            else:
                cmd = 'spfeas -i {} -o features --sect-size 100000 --block {} \
                    --band-positions {} {} {} --scales {} --triggers {}'

                if self._block_size > max(self._scales):
                    LOG.warning("The block size cannot be larger than scale,\
                                 skipping ...")
                    return
                scale_string = " ".join(map(str, self._scales))
                feature_string = " ".join(map(str, spfeas_features))
                spfeas = cmd.format(self._image_path, self._block_size,
                                    self._bands[0], self._bands[1],
                                    self._bands[2], scale_string,
                                    feature_string)
                LOG.info("Running spfeas command: {}".format(spfeas))
                os.system(spfeas)
        if rid_feature:
            LOG.info("Creating Road Intersection Density feature")
            folder = self.__get_folder_name(feature_names=rid_feature,
                                            scales=[max(self._scales)])
            if not os.path.exists(folder):
                os.mkdir(folder)

            path = os.path.join(folder, 'feature.rid')
            if os.path.exists(path):
                LOG.info("Feature already exists, skipping ...")
                return

            kernel = Kernel(road_width=15, road_length=50,
                            kernel_type=ktype.GAUSSIAN)
            intersections = RoadIntersections(self._image_path, kernel,
                                              peak_min_distance=100)
            rid = RoadIntersectionDensity(self._image_path, intersections, scale=80,
                                          block_size=self._block_size)
            rid.create()
            RoadIntersectionDensity.save(rid, path)

    def get(self):
        if self._feature:
            return self._feature

        spfeas_features = sorted(set(self._feature_names).
                                 intersection(['hog', 'lsr']))
        rid_feature = sorted(set(self._feature_names).intersection(['rid']))

        if spfeas_features:
            folder = self.__get_folder_name(feature_names=spfeas_features)
            feature_path = self.__get_feature_from_folder(folder)
            if feature_path is None:
                err = "Cannot find specified feature folder: {}".format(folder)
                raise IOError(err)

            spfeas_features = np.array(read_geotiff(feature_path))

        if rid_feature:
            folder = self.__get_folder_name(feature_names=rid_feature,
                                            scales=[max(self._scales)])
            feature_path = self.__get_feature_from_folder(folder)

            if feature_path is None:
                err = "Cannot find specified feature folder: {}".format(folder)
                raise IOError(err)

            rid = RoadIntersectionDensity.load(feature_path)
            rid_feature = rid.get()

        if len(spfeas_features) > 0 and len(rid_feature) > 0:
            self._feature = np.concatenate((spfeas_features, rid_feature),
                                           axis=0)
        elif len(spfeas_features) > 0:
            self._feature = spfeas_features
        elif len(rid_feature) > 0:
            self._feature = rid_feature
        return self._feature

    @property
    def block_size(self):
        return self._block_size

    @property
    def image_path(self):
        return self._image_path

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def scales(self):
        return self._scales
