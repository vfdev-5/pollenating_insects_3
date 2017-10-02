import numpy as np
import cv2
from imgaug import augmenters as iaa

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


test_geom_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
])

test_color_aug = iaa.Sequential([
    iaa.OneOf([
        iaa.Add((-25, 25), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    ])
])


def transform(x):
    # RGBA -> RGB
    if x.shape[2] == 4:
        x = x[:, :, 0:3]

    # Resize to 224x224
    x = cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # Define geom and color augmentations:
    x = train_geom_aug.augment_image(x)
    x = train_color_aug.augment_image(x)

    # 'RGB'->'BGR' and float32
    x = x[:, :, ::-1].astype(np.float32)
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x


def transform_test(x):
    # RGBA -> RGB
    if x.shape[2] == 4:
        x = x[:, :, 0:3]

    # Resize to 224x224
    x = cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # Define geom and color augmentations:
    x = test_geom_aug.augment_image(x)
    x = test_color_aug.augment_image(x)

    # 'RGB'->'BGR' and float32
    x = x[:, :, ::-1].astype(np.float32)
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x
