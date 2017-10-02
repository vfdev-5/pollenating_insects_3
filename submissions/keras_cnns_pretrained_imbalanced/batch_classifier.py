from collections import defaultdict

import imp

import types

from joblib import delayed
from joblib import Parallel

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam

from rampwf.workflows.image_classifier import _chunk_iterator, _to_categorical, get_nb_minibatches, BatchGeneratorBuilder


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

        # Setup val_gen_builder
        try:
            problem_file = 'problem.py'
            problem = imp.load_source('problem', problem_file)
            (folder, X_val), y_val = problem.get_test_data()
            val_gen_builder = BatchGeneratorBuilder(X_val, y_val,
                                                    train_gen_builder.transform_test_img,
                                                    train_gen_builder.transform_test_img,
                                                    folder,
                                                    train_gen_builder.chunk_size,
                                                    train_gen_builder.n_classes,
                                                    train_gen_builder.n_jobs)
            val_gen_builder.shuffle = True

            # Try to pickle transform method:
            try:
                import pickle
                pickle.dumps(val_gen_builder.transform_img)
                pickle.dumps(val_gen_builder.transform_test_img)
                val_gen_builder._get_generator = types.MethodType(local_get_generator2, val_gen_builder)
            except:
                val_gen_builder._get_generator = types.MethodType(local_get_generator, val_gen_builder)

            val_gen_builder.get_train_valid_generators = \
                types.MethodType(local_get_train_valid_generators, val_gen_builder)
            gen_val, nb_valid, _ = val_gen_builder.get_train_valid_generators(batch_size=batch_size)

        except Exception as e:
            print("ERROR:", e)
            gen_val = None
            nb_valid = None

        print("Train dataset size: {} | Validation dataset size: {}".format(nb_train, nb_valid))

        self.model.fit_generator(
            gen_train,
            steps_per_epoch=3 * get_nb_minibatches(nb_train, batch_size),
            epochs=60 // 3,
            max_queue_size=batch_size,
            callbacks=get_callbacks(self.model, 'data'),
            class_weight=class_weights,
            validation_data=gen_val,
            validation_steps=get_nb_minibatches(nb_train, batch_size) if nb_valid is not None else None,
            verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def _build_model(self):

        vgg16 = VGG16(include_top=False, weights='imagenet')
        for l in vgg16.layers:
            l.trainable = False
        inp = Input((224, 224, 3))
        x = vgg16(inp)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='linear', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(4096, activation='linear', name='fc2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        out = Dense(403, activation='softmax', name='predictions')(x)
        model = Model(inp, out)
        model.compile(
            loss='categorical_crossentropy', optimizer=Adam(lr=0.001),
            metrics=['accuracy', f170])
        model.name = "VGG16"
        return model


# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from operator import itemgetter
import json
from skimage.io import imread


def compute_class_undersampled_indices(input_path='data', n_clusters=15, hist_size=100):
    train_csv_path = os.path.join(input_path, 'train.csv')
    assert os.path.exists(train_csv_path)
    train_csv_df = pd.read_csv(train_csv_path)
    class_id_gb = train_csv_df.groupby('class')
    class_count = class_id_gb['class'].count()
    freq_classes = class_count[class_count > 250].sort_values()

    def compute_histogram(img, hist_size=100):
        hist = cv2.calcHist([img], [0], mask=None, histSize=[hist_size], ranges=(0, 255))
        hist = cv2.normalize(hist, dst=hist)
        return hist

    def get_filename(image_id, image_type):
        check_dir = False
        ext = ''
        prefix = ''
        if "Train" in image_type:
            data_path = os.path.join(input_path, 'imgs')
        else:
            raise Exception("Image type '%s' is not recognized" % image_type)

        if check_dir and not os.path.exists(data_path):
            os.makedirs(data_path)
        if len(ext) > 0:
            return os.path.join(data_path, "{}{}.{}".format(prefix, image_id, ext))
        return os.path.join(data_path, "{}{}".format(prefix, image_id, ext))

    def get_image_data(image_id, image_type):
        fname = get_filename(image_id, image_type)

        img = cv2.imread(fname)
        if img is None:
            img = imread(fname)
            assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    image_size = (224, 224)
    crop_size = 100
    n_features = 2 * hist_size
    center = (image_size[0] // 2, image_size[1] // 2)

    class_clusters = {}

    for k, freq_class in enumerate(freq_classes.index):
        freq_class_ids = class_id_gb.get_group(freq_class)['id'].values
        print("\n-- class: %i (%i/%i), n=%i \n" % (freq_class, k, len(freq_classes.index), len(freq_class_ids)))

        X = np.zeros((len(freq_class_ids), n_features), dtype=np.float32)
        for i, image_id in enumerate(freq_class_ids):
            if i % 100 == 0:
                print("--", i, "/", len(freq_class_ids))
            img = get_image_data(image_id, "Train")
            img = cv2.resize(img, dsize=image_size[::-1])

            # crop
            proc = img[center[1] - crop_size:center[1] + crop_size, center[0] - crop_size:center[0] + crop_size, :]
            # Blur
            proc = cv2.GaussianBlur(proc, (7, 7), 0)
            hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            sat = hsv[:, :, 1]
            hist_hue = compute_histogram(hue, hist_size)
            hist_sat = compute_histogram(sat, hist_size)
            X[i, 0:hist_size] = hist_hue[:, 0]
            X[i, hist_size:2 * hist_size] = hist_sat[:, 0]

        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        clusters = kmeans.predict(X)

        class_clusters[freq_class] = clusters

        n_images_per_cluster = int(255 / n_clusters)

    output = {}

    for freq_class in freq_classes.index:
        freq_class_ids = class_id_gb.get_group(freq_class)['id'].values
        clusters = class_clusters[freq_class]
        undersampled_class_indices = []

        # sort clusters by size in ascending order:
        ordered_cluster_indice_size = []
        for cluster_index in range(n_clusters):
            indices = np.where(clusters == cluster_index)[0]
            ordered_cluster_indice_size.append((cluster_index, len(indices), indices))

        ordered_cluster_indice_size = sorted(ordered_cluster_indice_size, key=itemgetter(1))

        for i, (cluster_index, s, indices) in enumerate(ordered_cluster_indice_size):
            np.random.shuffle(indices)
            k = min(n_images_per_cluster, len(indices))
            if i * n_images_per_cluster > len(undersampled_class_indices):
                k = min((i+1) * n_images_per_cluster - len(undersampled_class_indices), len(indices))
            undersampled_class_indices.extend(freq_class_ids[indices[:k]])

        output[freq_class] = undersampled_class_indices

    for freq_class in freq_classes.index:
        print("%i : n=%i" % (freq_class, len(output[freq_class])))
        diff = set(output[freq_class]) - set(class_id_gb.get_group(freq_class)['id'].values)
        assert len(diff) == 0, "WTF : {}: {}".format(freq_class, diff)

    return output


def load_class_undersampled_indices(path):
    out = {}
    with open(path, 'r') as f:
        d = json.load(f)
    for k in d:
        out[int(k)] = d[k]
    return out


def save_class_undersampled_indices(path, output):
    with open(path, 'w') as f:
        d = {}
        for k in output:
            d[str(k)] = np.array(output[k]).tolist()
        json.dump(d, f)


def load_or_compute_class_undersampled_indices(path='data'):
    path = os.path.join(path, 'class_undersampled_indices.json')
    if os.path.exists(path):
        return load_class_undersampled_indices(path)
    else:
        output = compute_class_undersampled_indices()
        save_class_undersampled_indices(path, output)
        return output


def local_get_train_valid_generators(self, batch_size=256, **kwargs):
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
        - nb_train is the number of training examples
    """
    class_weights = defaultdict(int)
    max_count = 0
    for class_index in self.y_array:
        class_weights[class_index] += 1
        if class_weights[class_index] > max_count:
            max_count = class_weights[class_index]

    for class_index in class_weights:
        class_weights[class_index] = max_count * 1.0 / class_weights[class_index]

    indices = np.arange(self.nb_examples)
    train_indices = indices
    gen_train = self._get_generator(indices=train_indices, batch_size=batch_size)

    return gen_train, self.nb_examples, class_weights


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
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from datetime import datetime


def get_callbacks(model, output_path='data'):

    save_prefix = model.name
    weights_path = os.path.join(output_path, "weights")
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    weights_filename = os.path.join(weights_path,
                                    save_prefix + "_{epoch:02d}_val_loss={val_loss:.4f}")

    metrics_names = list(model.metrics_names)
    metrics_names.remove('loss')
    for mname in metrics_names:
        weights_filename += "_val_%s={val_%s:.4f}" % (mname, mname)
    weights_filename += ".h5"

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss',
                                       save_best_only=False, save_weights_only=False)
    now = datetime.now()

    csv_logger = CSVLogger(os.path.join(weights_path,
                                        'training_%s_%s.log' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M")))))

    onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)

    callbacks = [model_checkpoint, csv_logger, onplateau]
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
