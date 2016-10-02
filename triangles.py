import cv2

import numpy as np
from numpy.linalg import norm
import os
import cPickle as pickle
from sklearn import metrics
from matplotlib.colors import Normalize
import random

import time

SZ = 20


class StatModel(object):
    def __init__(self):
        self.model = None

    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class ANN(StatModel):
    def __init__(self, hidden=20):
        super(ANN, self).__init__()
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
        self.model.setLayerSizes(np.array([400, hidden, 2]))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        predicted_labels = []

        for sample in samples:
            predicted_label = self.model.predict(np.array([sample.ravel()], dtype=np.float32))[0]
            predicted_labels.append(predicted_label)
        return predicted_labels


class SVM(StatModel):
    def __init__(self, C=1.0, gamma=0.5, class_weights=None):
        super(SVM, self).__init__()
        self.model = cv2.ml.SVM_create()
        self.model.setC(C)
        self.model.setGamma(gamma)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setClassWeights(class_weights)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


class LogisticRegression(StatModel):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.model = cv2.ml.LogisticRegression_create()
        self.model.setLearningRate(0.001)
        self.model.setIterations(1000)
        self.model.setTrainMethod(cv2.ml.LOGISTIC_REGRESSION_BATCH)
        self.model.setRegularization(1)
        self.model.setMiniBatchSize(1)

    def train(self, samples, responses):
        responses = np.float32(responses)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, samples, labels):
    resp = model.predict(samples)

    indices = []
    for i in range(0, len(labels)):
        if labels[i] != resp[i]:
            indices.append(i)

    confusion_matrix = metrics.confusion_matrix(labels, resp)
    mean_squared_error = metrics.mean_squared_error(labels, resp)
    accuracy = metrics.accuracy_score(labels, resp)
    precision = metrics.precision_score(labels, resp)
    f1 = metrics.f1_score(labels, resp)
    recall = metrics.recall_score(labels, resp)
    auc = metrics.roc_auc_score(labels, resp)

    score_metrics = {"mse": mean_squared_error, "accuracy": accuracy, "precision": precision,
                     "f1": f1, "recall": recall, "auc": auc}

    return confusion_matrix, score_metrics, indices


def print_model_evaluation(confusion_matrix, score_metrics):
    print_cm(confusion_matrix, ["non-triangle", "triangle"])
    print "Mean Squared Error: %f" % score_metrics["mse"]
    print "Accuracy: %f" % score_metrics["accuracy"]
    print "Precision: %f" % score_metrics["precision"]
    print "F1: %f" % score_metrics["f1"]
    print "Recall: %f" % score_metrics["recall"]
    print "AUC: %f" % score_metrics["auc"]


def load_raw_data(path, sub_folders):
    images = []
    labels = []
    label = -1

    for folder in sub_folders:
        sub_path = os.path.join(path, folder)
        label += 1
        for f in os.listdir(sub_path):
            if f.endswith(".png"):
                file_path = os.path.join(sub_path, f)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)


def dump_data(images, labels, filename):
    with open(filename, "wb") as f:
        pickle.dump(images, f)
        pickle.dump(labels, f)


def load_pickle_data(filename):
    with open(filename, "rb") as f:
        images = pickle.load(f)
        labels = pickle.load(f)

    return images, labels


def extract_features_simple(images):
    return np.float32(images.reshape(-1, SZ * SZ) / 255.0)


def extract_features(images):
    samples = []
    for img in images:
        img_hog = hog(img)
        # cnt_feature = 0 if number_of_contours(img) > 5 else 1

        features = img_hog
        samples.append(features)

    return np.float32(samples)


def number_of_contours(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    return len(contours)


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n * ang / (2 * np.pi))
    # bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
    # mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    bin_cells = \
        bin[:5, :5], bin[5:10, :5], bin[10:15, :5], bin[15:, :5], \
        bin[:5, 5:10], bin[5:10, 5:10], bin[10:15, 5:10], bin[15:, 5:10], \
        bin[:5, 10:15], bin[5:10, 10:15], bin[10:15, 10:15], bin[10:15], \
        bin[:5, 15:], bin[5:10, 15:], bin[10:15, 15:], bin[15:, 15:]

    mag_cells = \
        mag[:5, :5], mag[5:10, :5], mag[10:15, :5], mag[15:, :5], \
        mag[:5, 5:10], mag[5:10, 5:10], mag[10:15, 5:10], mag[15:, 5:10], \
        mag[:5, 10:15], mag[5:10, 10:15], mag[10:15, 10:15], mag[10:15], \
        mag[:5, 15:], mag[5:10, 15:], mag[10:15, 15:], mag[15:, 15:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist


class MidpointNormalize(Normalize):
    """Utility function to move the midpoint of a colormap to be around the values of interest."""

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def start(self, message):
        self.start_time = time.time()
        print message

    def stop(self):
        print "[INFO] finished in %0.3fs" % (time.time() - self.start_time)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrix"""
    column_width = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * column_width
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(column_width) % label,
    print
    # Print rows
    for i, label in enumerate(labels):
        print "    %{0}s".format(column_width) % label,
        for j in range(len(labels)):
            cell = "%{0}.1f".format(column_width) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


def get_split_indices(labels, n_triangles, n_non_triangles):
    triangle_indices = []
    non_triangle_indices = []

    triangle_label = 1
    for i, item in enumerate(labels):
        if item == triangle_label:
            triangle_indices.append(i)
        else:
            non_triangle_indices.append(i)

    random.shuffle(triangle_indices)
    random.shuffle(non_triangle_indices)

    triangle_train_indices = triangle_indices[:n_triangles]
    triangle_test_indices = triangle_indices[n_triangles:]
    non_triangle_train_indices = non_triangle_indices[:n_non_triangles]
    non_triangle_test_indices = non_triangle_indices[n_non_triangles:]

    train_indices = triangle_train_indices + non_triangle_train_indices
    test_indices = triangle_test_indices + non_triangle_test_indices

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return train_indices, test_indices
