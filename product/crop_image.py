from util import read_geotiff
from matplotlib import pyplot as plt
from satsense.bands import WORLDVIEW2
from satsense.image import SatelliteImage, get_rgb_bands
import numpy as np
import argparse


class Crop:
    coords = []
    image = None

    def __init__(self, filename):
        self.image = self.__get_rgb_image(filename)

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("button_press_event", self.__onclick)

    def __onclick(self, event):
        if event.button != 3:
            return
        ix, iy = int(event.xdata), int(event.ydata)
        self.coords.append((ix, iy))

        if len(self.coords) == 2:
            self.__crop()
            plt.close()

    def __get_rgb_image(self, filename):
        bands = WORLDVIEW2
        self.image = np.array(read_geotiff(filename))
        rgb = np.stack([self.image[4], self.image[2], self.image[1]], axis=2)
        rgb = rgb / np.max(rgb)
        return rgb

    def __crop(self):
        x1 = self.coords[0][0]
        x2 = self.coords[1][0]

        y1 = self.coords[0][1]
        y2 = self.coords[1][1]
        self.image = self.image[min([y1, y2]):max([y1, y2]), min([x1, x2]):max([x1, x2]), :]

    def crop_image(self):
        plt.imshow(self.image)
        plt.show()
        return self.image

    def save_rgb(self, filename, image):
        plt.imsave(filename, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop a TIFF image")
    parser.add_argument("filename",
                        help="The satellite TIF image to analyze")
    parser.add_argument("--save", help="Select the path where to save the\
                         cropped image.")
    parser.add_argument("--show", action='store_true', help="Show the cropped\
                        image.")

    args = parser.parse_args()

    crop = Crop(args.filename)
    image = crop.crop_image()

    if args.save:
        crop.save_rgb(args.save + ".png", image)

    if args.show:
        plt.imshow(image)
        plt.show()
