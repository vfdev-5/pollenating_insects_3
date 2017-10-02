from __future__ import print_function
from collections import defaultdict

import imp

import types
from imblearn.over_sampling import RandomOverSampler

from joblib import delayed
from joblib import Parallel
import numpy as np
import os

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

from rampwf.workflows.image_classifier import _chunk_iterator, _to_categorical, get_nb_minibatches


class BatchClassifier(object):
    def __init__(self):
        self.model = self._build_model()

    def fit(self, train_gen_builder):

        # Try to pickle transform method:
        try:
            import pickle
            pickle.dumps(train_gen_builder.transform_img)
            pickle.dumps(train_gen_builder.transform_test_img)
            train_gen_builder._get_generator = types.MethodType(local_get_generator2, train_gen_builder)
        except:
            print("Failed to pickle 'transform' function")
            train_gen_builder._get_generator = types.MethodType(local_get_generator, train_gen_builder)

        train_gen_builder.get_train_valid_generators = \
            types.MethodType(local_get_train_valid_generators, train_gen_builder)
        train_gen_builder.n_jobs = 8
        train_gen_builder.shuffle = True

        batch_size = 64
        train_gen_builder.chunk_size = batch_size * 5
        gen_train, nb_train, class_weights = train_gen_builder.get_train_valid_generators(batch_size=batch_size)

        gen_valid = None
        nb_valid = None

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=2,
            max_queue_size=batch_size,
            callbacks=get_callbacks(),
            class_weight=class_weights,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, batch_size) if nb_valid is not None else None,
            verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def _build_model(self):

        vgg16 = VGG16(include_top=False, weights='imagenet')

        layers_to_train = [
            'block5_conv1', 'block5_conv2', 'block5_conv3',
        ]
        for l in vgg16.layers:
            if l.name not in layers_to_train:
                l.trainable = False
        inp = Input((224, 224, 3))
        x = vgg16(inp)
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        out = Dense(403, activation='softmax', name='predictions')(x)
        model = Model(inp, out)
        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=0.001),
            metrics=['accuracy', f170])
        model.name = "VGG16"

        model.summary()
        return model


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================


def local_get_train_valid_generators(self, batch_size=256):
    """Build train and valid generators for keras.

    This method is used by the user defined `Classifier` to o build train
    and valid generators that will be used in keras `fit_generator`.

    Parameters
    ==========

    batch_size : int
        size of mini-batches

    Returns
    =======

    a 3-tuple (gen_train, nb_train, class_weights) where:
        - gen_train is a generator function for training data
        - gen_valid is a generator function for valid data
        - nb_train is the number of training examples
        - nb_valid is the number of validation examples
    The number of training and validation data are necessary
    so that we can use the keras method `fit_generator`.
    """

    # Oversample dataset
    rs = RandomOverSampler()

    self.X_array, self.y_array = rs.fit_sample(self.X_array[:, None], self.y_array)
    self.X_array = self.X_array.ravel()
    self.nb_examples = len(self.y_array)

    class_weights = defaultdict(int)
    max_count = 0
    for class_index in self.y_array:
        class_weights[class_index] += 1
        if class_weights[class_index] > max_count:
            max_count = class_weights[class_index]

    for class_index in class_weights:
        class_weights[class_index] = max_count * 1.0 / class_weights[class_index]

    nb_train = self.nb_examples
    indices = np.arange(self.nb_examples)
    gen_train = self._get_generator(indices=indices, batch_size=batch_size)

    return gen_train, nb_train, class_weights


def local_get_generator(self, indices=None, batch_size=256):
    if indices is None:
        indices = np.arange(self.nb_examples)
    # Infinite loop, as required by keras `fit_generator`.
    # However, as we provide the number of examples per epoch
    # and the user specifies the total number of epochs, it will
    # be able to end.

    y_stats = defaultdict(int)

    while True:

        # Display feeded dataset stats
        if len(y_stats) > 0:
            print("\n\n------------------------------------")
            print("Dataflow classes distribution : \n")
            for k in y_stats:
                print("'{}': {} |".format(str(k), y_stats[k]), end=' \t ')
            print("\n\n------------------------------------\n\n")

        if self.shuffle:
            np.random.shuffle(indices)
        it = _chunk_iterator(
            X_array=self.X_array[indices], folder=self.folder,
            y_array=self.y_array[indices], chunk_size=self.chunk_size,
            n_jobs=self.n_jobs)
        for X, y in it:

            # 1) Preprocessing of X and y
            # X = Parallel(n_jobs=self.n_jobs, backend='multiprocessing')(
            #         delayed(self.transform_img)(x) for x in X)
            X = np.array([self.transform_img(x) for x in X])
            # X is a list of numpy arrrays at this point, convert it to a
            # single numpy array.
            try:
                X = [x[np.newaxis, :, :, :] for x in X]
            except IndexError:
                # single channel
                X = [x[np.newaxis, np.newaxis, :, :] for x in X]
            X = np.concatenate(X, axis=0)
            X = np.array(X, dtype='float32')

            for class_index in y:
                y_stats[class_index] += 1

            # Convert y to onehot representation
            y = _to_categorical(y, num_classes=self.n_classes)

            # 2) Yielding mini-batches
            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size]


def _chunk_iterator2(parallel, X_array, folder, y_array=None, chunk_size=1024):
    from skimage.io import imread
    for i in range(0, len(X_array), chunk_size):
        X_chunk = X_array[i:i + chunk_size]
        filenames = [os.path.join(folder, '{}'.format(x)) for x in X_chunk]
        X = parallel(delayed(imread)(filename) for filename in filenames)
        if y_array is not None:
            y = y_array[i:i + chunk_size]
            yield X, y
        else:
            yield X


def local_get_generator2(self, indices=None, batch_size=256):
    if indices is None:
        indices = np.arange(self.nb_examples)
    # Infinite loop, as required by keras `fit_generator`.
    # However, as we provide the number of examples per epoch
    # and the user specifies the total number of epochs, it will
    # be able to end.

    y_stats = defaultdict(int)
    with Parallel(n_jobs=self.n_jobs, backend='threading') as parallel:
        while True:

            # Display feeded dataset stats
            if len(y_stats) > 0:
                print("\n\n------------------------------------")
                print("Dataflow classes distribution : \n")
                for k in y_stats:
                    print("'{}': {} |".format(str(k), y_stats[k]), end=' \t ')
                print("\n\n------------------------------------\n\n")

            if self.shuffle:
                np.random.shuffle(indices)
            it = _chunk_iterator2(parallel,
                                  X_array=self.X_array[indices], folder=self.folder,
                                  y_array=self.y_array[indices], chunk_size=self.chunk_size)

            for X, y in it:
                # 1) Preprocessing of X and y
                X = parallel(delayed(self.transform_img)(x) for x in X)
                # X = np.array([self.transform_img(x) for x in X])
                # X is a list of numpy arrrays at this point, convert it to a
                # single numpy array.
                try:
                    X = [x[np.newaxis, :, :, :] for x in X]
                except IndexError:
                    # single channel
                    X = [x[np.newaxis, np.newaxis, :, :] for x in X]
                X = np.concatenate(X, axis=0)
                X = np.array(X, dtype='float32')

                for class_index in y:
                    y_stats[class_index] += 1

                # Convert y to onehot representation
                y = _to_categorical(y, num_classes=self.n_classes)

                # 2) Yielding mini-batches
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size], y[i:i + batch_size]


from keras import backend as K
from keras.callbacks import ReduceLROnPlateau


def get_callbacks():
    onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    callbacks = [onplateau, ]
    return callbacks


def f170(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    numer = 2.0 * true_positives
    denom = predicted_positives + possible_positives + K.epsilon()
    f1 = numer / denom
    score = K.sum(K.round(K.clip((f1 - 0.7)*10.0, 0, 1)), axis=0) / K.int_shape(f1)[0]
    return score
