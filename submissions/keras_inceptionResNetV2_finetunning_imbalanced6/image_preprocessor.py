import numpy as np
import cv2
from imgaug import augmenters as iaa

from keras.applications.imagenet_utils import preprocess_input 

SIZE = (299, 299)  # (w, h)

train_geom_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Sometimes(0.5, (iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-5, 5),
        order=3,
        mode='edge'
    ))),
])

train_color_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    ])

])

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


def _transform(x, geom_aug=None, color_aug=None):
    # RGBA -> RGB
    if x.shape[2] == 4:
        x = x[:, :, 0:3]

    # Resize to SIZE
    x = cv2.resize(x, dsize=SIZE, interpolation=cv2.INTER_CUBIC)

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
