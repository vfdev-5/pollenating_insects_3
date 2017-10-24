import numpy as np
import cv2
from imgaug import augmenters as iaa
from imgaug.augmenters.meta import Augmenter

from keras.applications.imagenet_utils import preprocess_input

SIZE = (340, 340)  # (w, h)


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        self.size = (int(size), int(size))
        self.padding = padding

    def __call__(self, img):

        h, w, _ = img.shape
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)
        return img[y1:y1 + th, x1:x1 + tw, :]


train_geom_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Fliplr(1.0),
        iaa.Flipud(1.0),
        iaa.Affine(
            rotate=(0, 360),
            order=3,
            mode='edge'
        )])

])

# train_color_aug = iaa.Sequential([
#     iaa.OneOf([
#         iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
#         iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
#         iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
#     ])
#
# ])
train_color_aug = None


# test_geom_aug = iaa.Sequential([
#     iaa.Fliplr(0.5),
#     iaa.Flipud(0.5),
# ])
test_geom_aug = None


# test_color_aug = iaa.Sequential([
#     iaa.OneOf([
#         iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
#         iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
#         iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
#     ])
# ])
test_color_aug = None


random_crop = RandomCrop(299)


def _transform(x, geom_aug=None, color_aug=None):
    # RGBA -> RGB
    if x.shape[2] == 4:
        x = x[:, :, 0:3]

    # Resize to SIZE
    x = cv2.resize(x, dsize=SIZE, interpolation=cv2.INTER_CUBIC)

    # Crop to 299
    x = random_crop(x)

    # Data augmentation:
    if geom_aug is not None:
        x = geom_aug.augment_image(x)
    if color_aug is not None:
        x = color_aug.augment_image(x)

    # to float32
    x = x.astype(np.float32)
    x = preprocess_input(x, data_format='channels_last')
    return x


def transform(x):
    return _transform(x, train_geom_aug, train_color_aug)


def transform_test(x):
    return _transform(x, test_geom_aug, test_color_aug)
