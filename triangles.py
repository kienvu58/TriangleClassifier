import cv2

import numpy as np
from numpy.linalg import norm
import os
import cPickle as pickle
from sklearn import metrics
import helper
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import time

SZ = 20  # size of each image is SZ x SZ
TRAIN_DATA_PATH = "../data/train/"
SUB_FOLDERS = ["non-triangle", "triangle"]
DATA_FN = "images.dat"


class StatModel(object):
    def __init__(self):
        self.model = None

    def load(self, fn):
        self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


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
    helper.print_cm(confusion_matrix, ["non-triangle", "triangle"])
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


def load_data():
    if not os.path.isfile(DATA_FN):
        images, labels = load_raw_data(TRAIN_DATA_PATH, SUB_FOLDERS)
        dump_data(images, labels, DATA_FN)
    else:
        images, labels = load_pickle_data(DATA_FN)

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
        print "finished in %0.3f s" % (time.time() - self.start_time)


def main():
    timer = Timer()
    timer.start("loading data...")
    raw_images, labels = load_data()
    timer.stop()

    timer.start("preprocessing...")
    images_1 = cv2.GaussianBlur(raw_images, (3, 3), 0)
    _, images_2 = cv2.threshold(images_1, 50, 255, cv2.THRESH_TOZERO)
    images_3 = cv2.medianBlur(images_2, 3)

    # samples = extract_features_simple(images)
    samples = extract_features(images_3)

    # chunk data
    n = 1000

    data = zip(samples, labels)
    triange_index = range(100300, 110300)
    non_triange_index = range(100000)

    trianges = np.random.choice(triange_index, n)
    non_trianges = np.random.choice(non_triange_index, n)

    triangles_sample, triangles_label = samples[trianges], labels[trianges]
    non_triangles_sample, non_triangles_label = samples[non_trianges], labels[non_trianges]

    balanced_samples = np.concatenate([triangles_sample, non_triangles_sample])
    balanced_labels = np.concatenate([triangles_label, non_triangles_label])

    # shuffle data
    rand = np.random.RandomState(3426)
    shuffle = rand.permutation(len(samples))
    samples, labels = samples[shuffle], labels[shuffle]

    images_1 = images_1[shuffle]
    images_2 = images_2[shuffle]
    images_3 = images_3[shuffle]
    raw_images = raw_images[shuffle]

    # split train, test
    train_n = int(0.8 * len(samples))
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    raw_images_train, raw_image_test = np.split(raw_images, [train_n])

    timer.stop()

    # timer.start("training SVM with class weights...")
    # model = SVM(C=100, gamma=1, class_weights=np.array([0.1, 1.0]))
    # model.train(samples_train, labels_train)
    # timer.stop()

    timer.start("training SVM with balanced samples...")
    model = SVM(C=10, gamma=6)
    model.train(balanced_samples, balanced_labels)
    timer.stop()

    # timer.start("training LogisticRegression model...")
    # model = LogisticRegression()
    # model.train(balanced_samples, balanced_labels)
    # timer.stop()

    timer.start("evaluate model...")
    confusion_matrix, score_metrics, indices = evaluate_model(model, samples, labels)
    print_model_evaluation(confusion_matrix, score_metrics)
    timer.stop()

    print len(indices)
    for i in range(1, 101):
        plt.subplot(20, 20, 4*i-3)
        plt.imshow(raw_images[indices[i]], interpolation="nearest")
        plt.axis("off")
    for i in range(1, 101):
        plt.subplot(20, 20, 4*i-2)
        plt.imshow(images_1[indices[i]], interpolation="nearest")
        plt.axis("off")
    for i in range(1, 101):
        plt.subplot(20, 20, 4*i-1)
        plt.imshow(images_2[indices[i]], interpolation="nearest")
        plt.axis("off")
    for i in range(1, 101):
        plt.subplot(20, 20, 4*i)
        plt.imshow(images_3[indices[i]], interpolation="nearest")
        plt.axis("off")
    plt.show()

    # timer.start("sweeping parameters...")
    # C_range = np.logspace(-3, 5, 9)
    # gamma_range = np.logspace(-3, 3, 7)
    # classifiers = []
    # for C in C_range:
    #     for gamma in gamma_range:
    #         # train with balanced data
    #         clf = SVM(C=C, gamma=gamma)
    #         clf.train(balanced_samples, balanced_labels)
    #
    #         # train with skewed data
    #         # clf = SVM(C=C, gamma=gamma, class_weights=np.array([0.1, 1.0]))
    #         # clf.train(samples_train, labels_train)
    #
    #         classifiers.append((C, gamma, clf))
    #
    # criteria = ["precision", "accuracy", "f1"]
    # scores = {criterion: [] for criterion in criteria}
    # cm_list = []
    # sm_list = []
    # for clf in classifiers:
    #     confusion_matrix, score_metrics, _ = evaluate_model(clf[2], samples_test, labels_test)
    #     cm_list.append(confusion_matrix)
    #     sm_list.append(score_metrics)
    #     for criterion in criteria:
    #         scores[criterion].append(score_metrics[criterion])
    # timer.stop()
    #
    # max_index = scores[criteria[0]].index(max(scores[criteria[0]]))
    # C_index = max_index / len(gamma_range)
    # gamma_index = max_index % len(gamma_range)
    # print "the best parameters by %s: C=%f gamma=%f" % (criteria[0], C_range[C_index], gamma_range[gamma_index])
    # print_model_evaluation(cm_list[max_index], sm_list[max_index])
    #
    # # draw color map
    # for criterion in criteria:
    #     scores_by_criterion = np.array(scores[criterion]).reshape(len(C_range), len(gamma_range))
    #     plt.figure(figsize=(8, 6))
    #     plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    #     plt.imshow(scores_by_criterion, interpolation="nearest", cmap=plt.cool(),
    #                norm=MidpointNormalize(vmin=0.5, midpoint=0.92))
    #     plt.xlabel("gamma")
    #     plt.ylabel("C")
    #     plt.colorbar()
    #     plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    #     plt.yticks(np.arange(len(C_range)), C_range)
    #     plt.title("Validation " + criterion)
    #     plt.savefig(criterion + ".png")
    #
    # plt.show()


if __name__ == "__main__":
    main()
