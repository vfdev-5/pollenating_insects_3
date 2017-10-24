from __future__ import print_function
import os
from collections import defaultdict
from datetime import datetime

import types
from sklearn.model_selection import StratifiedShuffleSplit

from joblib import delayed
from joblib import Parallel
import numpy as np

import cv2

from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.losses import categorical_crossentropy


from rampwf.workflows.image_classifier import _chunk_iterator, _to_categorical, get_nb_minibatches

SIZE = (451, 451)
SEED = 2017

SUBMIT_NAME = os.path.basename(os.path.dirname(__file__))


class BatchClassifier(object):
    def __init__(self):
        self.model = self._build_model()

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s_%s' % (SUBMIT_NAME, now.strftime("%Y%m%d_%H%M"))

    def fit(self, train_gen_builder):

        valid_ratio = 0.1
        n_epochs = 10

        if 'LOCAL_TESTING' in os.environ:
            print("\n\n------------------------------")
            print("-------- LOCAL TESTING -------")
            print("------------------------------\n\n")
            if 'LOAD_BEST_MODEL' in os.environ:
                load_pretrained_model(self.model, self.logs_path)
                return

        # Try to pickle transform method:
        try:
            import pickle
            pickle.dumps(train_gen_builder.transform_img)
            pickle.dumps(train_gen_builder.transform_test_img)
            train_gen_builder._get_generator = types.MethodType(local_get_generator2, train_gen_builder)
        except Exception:
            print("Failed to pickle 'transform' function")
            train_gen_builder._get_generator = types.MethodType(local_get_generator, train_gen_builder)

        train_gen_builder.get_train_valid_generators = \
            types.MethodType(local_get_train_valid_generators, train_gen_builder)
        train_gen_builder.n_jobs = 8
        train_gen_builder.shuffle = True

        batch_size = 4
        train_gen_builder.chunk_size = batch_size * 8

        gen_train, gen_valid, nb_train, nb_valid, class_weights = \
            train_gen_builder.get_train_valid_generators(batch_size=batch_size,
                                                         valid_ratio=valid_ratio)

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        self._compile_model(self.model, lr=0.0001)
        self.model.summary()

        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=n_epochs,
            max_queue_size=batch_size,
            callbacks=get_callbacks(self.model, self.logs_path),
            class_weight=None,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, batch_size) if nb_valid is not None else None,
            verbose=1)

        # Load best trained model:
        load_pretrained_model(self.model, self.logs_path)

    def predict_proba(self, X):
        #n = 15
        #y_probas = []
        #for i in range(n):
        #    theta = i * 180.0 / n
        #    X_aug = np.zeros_like(X)
        #    if i > 0:
        #        for j, x in enumerate(X):
        #            X_aug[j, ...] = rotate(x, theta)
        #    else:
        #        X_aug = X
        #    y_probas.append(self.model.predict(X_aug))
        #y_proba = np.mean(np.array(y_probas), axis=0)
        #return y_proba
        return self.model.predict(X)

    def _compile_model(self, model, lr):
        loss = categorical_crossentropy
        model.compile(
            loss=loss, optimizer=Adam(lr=lr),
            metrics=['accuracy', f170])

    def _build_model(self):
        inceptionResNet = InceptionResNetV2(input_shape=SIZE + (3, ), include_top=False, weights='imagenet')
        x = inceptionResNet.outputs[0]
        
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.5)(x)
        out = Dense(403, activation='softmax', name='predictions')(x)

        model = Model(inceptionResNet.inputs, out)
        model.name = "InceptionResNetV2"
        return model


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================

def rotate(img, theta):
    center = (img.shape[1::-1] * np.array((0.5, 0.5))) - 0.5
    transform_matrix = cv2.getRotationMatrix2D(tuple(center), theta, 1.0)
    return cv2.warpAffine(img, transform_matrix, img.shape[1::-1],
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def local_get_train_valid_generators(self, batch_size=256, valid_ratio=0.3):
    """Build train and valid generators for keras.

    This method is used by the user defined `Classifier` to o build train
    and valid generators that will be used in keras `fit_generator`.

    Parameters
    ==========

    batch_size : int
        size of mini-batches
    valid_ratio : float between 0 and 1
        ratio of validation data

    Returns
    =======

    a 5-tuple (gen_train, gen_valid, nb_train, nb_valid, class_weights) where:
        - gen_train is a generator function for training data
        - gen_valid is a generator function for valid data
        - nb_train is the number of training examples
        - nb_valid is the number of validation examples
        - class_weights
    The number of training and validation data are necessary
    so that we can use the keras method `fit_generator`.
    """

    if valid_ratio > 0.0:
        ssplit = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=SEED)

        train_indices, valid_indices = next(ssplit.split(self.X_array, self.y_array))
        nb_train = len(train_indices)
        nb_valid = len(valid_indices)

        gen_train = self._get_generator(indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(indices=valid_indices, batch_size=batch_size)
    else:
        train_indices = np.arange(self.nb_examples)
        gen_train = self._get_generator(indices=train_indices, batch_size=batch_size)
        nb_train = len(train_indices)
        gen_valid = None
        nb_valid = None

    class_weights = defaultdict(int)
    max_count = 0
    for class_index in self.y_array[train_indices]:
        class_weights[class_index] += 1
        if class_weights[class_index] > max_count:
            max_count = class_weights[class_index]
    for class_index in class_weights:
        class_weights[class_index] = 1.0 + (max_count / class_weights[class_index])

    return gen_train, gen_valid, nb_train, nb_valid, class_weights


def local_get_generator(self, indices=None, batch_size=256):
    if indices is None:
        indices = np.arange(self.nb_examples)
    # Infinite loop, as required by keras `fit_generator`.
    # However, as we provide the number of examples per epoch
    # and the user specifies the total number of epochs, it will
    # be able to end.

    y_stats = defaultdict(int)
    np.random.seed(SEED)

    while True:

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
    np.random.seed(SEED)

    with Parallel(n_jobs=self.n_jobs, backend='threading') as parallel:
        while True:

            # Display feeded dataset stats
            # if len(y_stats) > 0:
            #     print("\n\n------------------------------------")
            #     print("Dataflow classes distribution : \n")
            #     for k in y_stats:
            #         print("'{}': {} |".format(str(k), y_stats[k]), end=' \t ')
            #     print("\n\n------------------------------------\n\n")

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


# ================================================================================================================
# =============== Keras callbacks and metrics ====================================================================

def step_decay(epoch, model, base=0.5, period=50, verbose=False):
    lr = K.get_value(model.optimizer.lr)
    factor = base if epoch > 0 and epoch % period == 0 else 1.0
    new_lr = lr * factor
    if verbose:
        print("New learning rate: %f" % new_lr)
    return new_lr


def get_callbacks(model, logs_path):

    callbacks = []
    # On plateau reduce LR callback measured on val_loss:
    onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    callbacks.append(onplateau)

    # LR schedule: step decay
    step_decay_f = lambda epoch: step_decay(epoch, model=model, base=0.2, period=3, verbose=True)
    lrscheduler = LearningRateScheduler(step_decay_f)
    callbacks.append(lrscheduler)

    # Store best weights, measured on val_loss
    save_prefix = model.name
    weights_path = os.path.join(logs_path, "weights")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    weights_filename = os.path.join(weights_path, save_prefix + "_best_val_loss.h5")

    model_checkpoint = ModelCheckpoint(weights_filename,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       save_weights_only=False)
    callbacks.append(model_checkpoint)

    # Some other callback on local testing
    if 'LOCAL_TESTING' in os.environ:
        from keras.callbacks import TensorBoard, CSVLogger

        csv_logger = CSVLogger(os.path.join(weights_path, 'training_%s.log' % (save_prefix)))
        callbacks.append(csv_logger)

        tboard = TensorBoard('logs', write_grads=True)
        callbacks.append(tboard)

    return callbacks


def false_negatives(y_true, y_pred):
    return K.mean(K.round(K.clip(y_true - y_pred, 0, 1)))


def categorical_crossentropy_with_f1(y_true, y_pred, a=2.0):
    return categorical_crossentropy(y_true, y_pred) + a * (1.0 - K.mean(f1(y_true, y_pred), axis=-1))


def f1(y_true, y_pred):
    # implicit thresholding at 0.5
    y_pred = K.round(K.clip(y_pred, 0, 1))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    numer = 2.0 * true_positives
    denom = predicted_positives + possible_positives + K.epsilon()
    f1 = numer / denom
    return f1


def f170(y_true, y_pred):
    score = f1(y_true, y_pred)
    score = K.sum(K.round(K.clip((score - 0.7) * 10.0, 0, 1)), axis=0) / K.int_shape(score)[0]
    return score

# ================================================================================================================
# =============== Other useful tools =============================================================================


def load_pretrained_model(model, logs_path):

    best_weights_filename = os.path.join(logs_path, "weights", "%s_best_val_loss.h5" % model.name)
    print("Load best loss weights: ", best_weights_filename)
    model.load_weights(best_weights_filename)
