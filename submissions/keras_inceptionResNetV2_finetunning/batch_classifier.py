from __future__ import print_function
import os
from collections import defaultdict
from datetime import datetime
from glob import glob

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
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import categorical_crossentropy

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


from rampwf.workflows.image_classifier import _chunk_iterator, _to_categorical, get_nb_minibatches


class BatchClassifier(object):
    def __init__(self):
        self.model = self._build_model()

        if 'LOGS_PATH' in os.environ:
            self.logs_path = os.environ['LOGS_PATH']
        else:
            now = datetime.now()
            self.logs_path = 'logs_%s' % now.strftime("%Y-%m-%d-%H-%M")

    def fit(self, train_gen_builder):

        has_pretraining = False
        valid_ratio = 0.3

        if 'LOCAL_TESTING' in os.environ:
            print("\n\n------------------------------")
            print("-------- LOCAL TESTING -------")
            print("------------------------------\n\n")
            n_epochs = 50
            
            if 'LOAD_BEST_MODEL' in os.environ:
                load_pretrained_model(self.model, self.logs_path)
                return            
        else:
            n_epochs = 50

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

        batch_size = 64
        train_gen_builder.chunk_size = batch_size * 8

        gen_train, gen_valid, nb_train, nb_valid, class_weights = \
            train_gen_builder.get_train_valid_generators(batch_size=batch_size,
                                                         valid_ratio=valid_ratio)

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        # Pretraining step
        if has_pretraining:
            layers_to_train = [                
                'predictions'
            ]
            for l in self.model.layers:
                l.trainable = False
                for ltt in layers_to_train:
                    if ltt in l.name:
                        l.trainable = True
                        break
            self._compile_model(self.model, lr=0.05)
            self.model.summary()

            self.model.fit_generator(
                gen_train,
                steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
                epochs=10,
                max_queue_size=batch_size,
                callbacks=get_callbacks(self.model, self.logs_path),
                class_weight=class_weights,
                validation_data=gen_valid,
                validation_steps=get_nb_minibatches(nb_valid, batch_size) if nb_valid is not None else None,
                verbose=1
            )
        
        
        # Finetunning
        layers_to_train = [
            'block8',            
            'conv_7b', 
            'predictions'
        ]
        for l in self.model.layers:
            l.trainable = False
            for ltt in layers_to_train:
                if ltt in l.name:
                    l.trainable = True
                    break
        
        self._compile_model(self.model, lr=0.001)
        self.model.summary()

        self.model.fit_generator(
            gen_train,
            steps_per_epoch=get_nb_minibatches(nb_train, batch_size),
            epochs=n_epochs,
            max_queue_size=batch_size,
            callbacks=get_callbacks(self.model, self.logs_path),
            class_weight=class_weights,
            validation_data=gen_valid,
            validation_steps=get_nb_minibatches(nb_valid, batch_size) if nb_valid is not None else None,
            verbose=1)

        # Load best trained model:
        load_pretrained_model(self.model, self.logs_path)

    def predict_proba(self, X):
        X_aug1 = np.zeros_like(X)
        X_aug2 = np.zeros_like(X)
        for i, x in enumerate(X):
            X_aug1[i, ...] = cv2.flip(x, 0)
            X_aug2[i, ...] = cv2.flip(x, 1)
        y_proba0 = self.model.predict(X)
        y_proba1 = self.model.predict(X_aug1)
        y_proba2 = self.model.predict(X_aug2)
        y_proba = 0.33 * (y_proba0 + y_proba1 + y_proba2)
        return y_proba

    def _compile_model(self, model, lr):
        loss = categorical_crossentropy
        model.compile(
            loss=loss, optimizer=Adam(lr=lr),
            metrics=['accuracy', f170])

    def _build_model(self):
        incResNetV2 = InceptionResNetV2(input_shape=(299, 299, 3), include_top=False, weights='imagenet')
        x = incResNetV2.outputs[0]
        
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(0.8)(x)
        out = Dense(403, activation='softmax', name='predictions')(x)
                
        model = Model(incResNetV2.inputs, out)
        model.name = "InceptionResNetV2"
        return model


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================


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
        ssplit = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio)

        train_indices, valid_indices = next(ssplit.split(self.X_array, self.y_array))
        gen_valid = self._get_generator(indices=valid_indices, batch_size=batch_size)
        nb_valid = len(valid_indices)

    else:
        train_indices = np.arange(self.nb_examples)
        gen_valid = None
        nb_valid = None

    class_weights = defaultdict(int)
    max_count = 0
    for class_index in self.y_array[train_indices]:
        class_weights[class_index] += 1
        if class_weights[class_index] > max_count:
            max_count = class_weights[class_index]
    for class_index in class_weights:
        class_weights[class_index] = np.log(1.0 + (max_count / class_weights[class_index]) ** 2)

    # Under+Oversample training data:
    # - undersample randomly images that count is larger a threshold
    # - oversample randomly all images

    undersampling_threshold = 350

    class_counts = np.zeros((403, ), dtype=np.int)
    for class_index in self.y_array[train_indices]:
        class_counts[class_index] += 1

    classes_to_undersample = np.where(class_counts > undersampling_threshold)[0]

    train_indices_to_undersample = [index for index in train_indices if self.y_array[index] in classes_to_undersample]
    train_indices_to_oversample = [index for index in train_indices if
                                   self.y_array[index] not in classes_to_undersample]

    rs = RandomUnderSampler()

    train_indices_undersampled, new_y_array = rs.fit_sample(np.array(train_indices_to_undersample)[:, None],
                                                            self.y_array[train_indices_to_undersample])
    rs = RandomOverSampler()
    new_train_indices = np.concatenate((train_indices_undersampled, np.array(train_indices_to_oversample)[:, None]))
    new_y_array = np.concatenate((new_y_array, self.y_array[train_indices_to_oversample]))

    new_train_indices, _ = rs.fit_sample(new_train_indices, new_y_array)
    new_train_indices = new_train_indices.ravel()
    gen_train = self._get_generator(indices=new_train_indices, batch_size=batch_size)
    nb_train = len(new_train_indices)

    self.nb_examples = nb_train + (0 if nb_valid is None else nb_valid)

    return gen_train, gen_valid, nb_train, nb_valid, class_weights


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
        # if len(y_stats) > 0:
        #     print("\n\n------------------------------------")
        #     print("Dataflow classes distribution : \n")
        #     for k in y_stats:
        #         print("'{}': {} |".format(str(k), y_stats[k]), end=' \t ')
        #     print("\n\n------------------------------------\n\n")

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


def get_callbacks(model, logs_path):

    # On plateau reduce LR callback measured on val_loss:
    onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    callbacks = [onplateau, ]

    # Store best weights, measured on val_loss
    save_prefix = model.name
    weights_path = os.path.join(logs_path, "weights")
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    weights_filename = os.path.join(weights_path, save_prefix + "_{epoch:02d}_val_loss={val_loss:.4f}")
    metrics_names = list(model.metrics_names)
    metrics_names.remove('loss')
    for mname in metrics_names:
        weights_filename += "_val_%s={val_%s:.4f}" % (mname, mname)
    weights_filename += ".h5"

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', save_best_only=True, save_weights_only=False)
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
    score = K.sum(K.round(K.clip((score - 0.7)*10.0, 0, 1)), axis=0) / K.int_shape(score)[0]
    return score


def load_pretrained_model(model, logs_path):
    weights_files = []
    weights_files.extend(glob(os.path.join(logs_path, "weights", "%s*.h5" % model.name)))
    assert len(weights_files) > 0, "Failed to load weights"
    best_weights_filename, best_val_loss = find_best_weights_file(weights_files, field_name='val_loss')
    print("Load best loss weights: ", best_weights_filename, best_val_loss)
    model.load_weights(best_weights_filename)


def find_best_weights_file(weights_files, field_name='val_loss', best_min=True):

    if best_min:
        best_value = 1e5
        comp = lambda a, b: a > b
    else:
        best_value = -1e5
        comp = lambda a, b: a < b

    if '=' != field_name[-1]:
        field_name += '='

    best_weights_filename = ""
    for f in weights_files:
        index = f.find(field_name)
        index += len(field_name)
        assert index >= 0, "Field name '%s' is not found in '%s'" % (field_name, f)
        end = f.find('_', index)
        val = float(f[index:end])
        if comp(best_value, val):
            best_value = val
            best_weights_filename = f
    return best_weights_filename, best_value

