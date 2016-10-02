from triangles import *
from lenet import LeNet
from keras.optimizers import SGD
from keras.utils import np_utils
import argparse

TRAIN_DATA_PATH = "../data/train/"
SUB_FOLDERS = ["non-triangle", "triangle"]
DATA_FN = "images.dat"


def load_data():
    if not os.path.isfile(DATA_FN):
        images, labels = load_raw_data(TRAIN_DATA_PATH, SUB_FOLDERS)
        dump_data(images, labels, DATA_FN)
    else:
        images, labels = load_pickle_data(DATA_FN)

    return images, labels


def train(ann, data, epochs=1):
    data_size = len(data)
    for x in xrange(epochs):
        counter = 0
        for training_input, training_output in data:
            if counter > data_size:
                break
            if counter % 1000 == 0:
                print "Epoch %d: Trained %d/%d" % (x, counter, data_size)
            counter += 1
            ann.train(np.array([training_input.ravel()], dtype=np.float32), np.array([training_output.ravel()], dtype=np.float32))
        print "Epoch %d complete" % x
    return ann


def evaluate_lenet(model, test_images, test_labels):
    predictions = model.predict(test_images, batch_size=128, verbose=1)
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    expected_labels = [np.argmax(test_label) for test_label in test_labels]

    confusion_matrix = metrics.confusion_matrix(expected_labels, predicted_labels)
    accuracy = metrics.accuracy_score(expected_labels, predicted_labels)
    precision = metrics.precision_score(expected_labels, predicted_labels)
    f1 = metrics.f1_score(expected_labels, predicted_labels)
    recall = metrics.recall_score(expected_labels, predicted_labels)

    print_cm(confusion_matrix, ["non-triangle", "triangle"])
    print "Accuracy: %f" % accuracy
    print "Precision: %f" % precision
    print "F1: %f" % f1
    print "Recall: %f" % recall


def main(args):
    timer = Timer()

    timer.start("[INFO] loading data...")
    images, labels = load_data()

    n_triangles_train = 2000
    n_non_triangles_train = 20000
    train_indices, test_indices = get_split_indices(labels, n_triangles_train, n_non_triangles_train)

    train_images, train_labels = images[train_indices], labels[train_indices]
    test_images, test_labels = images[test_indices], labels[test_indices]
    train_images = train_images[:, np.newaxis, :, :] / 255.0
    test_images = test_images[:, np.newaxis, :, :] / 255.0
    train_labels = np_utils.to_categorical(train_labels, 2)
    test_labels = np_utils.to_categorical(test_labels, 2)
    timer.stop()

    timer.start("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet.build(width=20, height=20, depth=1, classes=2,
                        weightsPath=args["weights"] if args["load_model"] > 0 else None)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    timer.stop()

    if args["load_model"] < 0:
        timer.start("[INFO] training...")
        model.fit(train_images, train_labels, batch_size=128, nb_epoch=20, verbose=1)
        timer.stop()

        # timer.start("[INFO] evaluating...")
        # loss, accuracy = model.evaluate(test_images, test_labels, batch_size=128, verbose=1)
        # print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
        # timer.stop()

    if args["save_model"] > 0:
        timer.start("[INFO] dumping weights to file...")
        model.save_weights(args["weights"], overwrite=True)
        timer.stop()

    timer.start("[INFO] evaluating...")
    evaluate_lenet(model, test_images, test_labels)
    timer.stop()


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save-model", type=int, default=-1,
                    help="(optional) whether or not model should be saved to disk")
    ap.add_argument("-l", "--load-model", type=int, default=-1,
                    help="(optional) whether or not pre-trained model should be loaded")
    ap.add_argument("-w", "--weights", type=str,
                    help="(optional) path to weights file")
    # main(vars(ap.parse_args()))
    main(vars(ap.parse_args(["--save-model", "1", "--weights", "output/lenet_weights.hdf5"])))