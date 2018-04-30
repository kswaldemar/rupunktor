import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rupunktor.converter import PUNKT_TAGS


def pickle_save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def train_test_split(x, y=None, test_ratio=0.2):
    """ Returns x_train, y_train, x_test, y_test """
    train_size = int(len(x) * (1.0 - test_ratio))
    if y is not None:
        return x[:train_size], y[:train_size], x[train_size:], y[train_size:]
    else:
        return x[:train_size], x[train_size:]


# Borrowed from https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
def print_confusion_matrix(confusion_matrix, class_names, figsize=(7, 5), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, cmap=sns.light_palette('navy'))
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    return fig


def confusion_matrix(y_true, y_pred, normalize=False):
    # Normal vector from categorical
    y_true = np.argmax(y_true, -1)
    y_pred = np.argmax(y_pred, -1)
    # Assuming it starts from 0
    num_classes = np.max(y_true) + 1
    mt = np.zeros((num_classes, num_classes))
    yt = []
    yp = []
    for i in range(num_classes):
        yt.append(y_true == i)
        yp.append(y_pred == i)

    for i in range(num_classes):
        for j in range(num_classes):
            mt[i, j] = np.sum(yp[i] * yt[j])
            if normalize:
                mt[i, j] /= np.sum(yp[i])
    return mt


def _prf(true_positives, false_positives, false_negatives):
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2. * precision * recall / (precision + recall)
    return precision * 100, recall * 100, f_score * 100


def show_statistics(conf_matrix):
    scores = []
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    for i in range(0, conf_matrix.shape[0]):
        true_positives = conf_matrix[i, i]
        false_positives = np.sum(conf_matrix, axis=1)[i] - true_positives
        false_negatives = np.sum(conf_matrix, axis=0)[i] - true_positives

        if i > 0:
            # Do not consider space
            overall_tp += true_positives
            overall_fp += false_positives
            overall_fn += false_negatives

        precision, recall, f_score = _prf(true_positives, false_positives, false_negatives)
        scores.append('{:<16} {:<9.3f} {:<9.3f} {:<9.3f}'.format(PUNKT_TAGS[i], precision, recall, f_score))
    print('=' * 46)
    print('{:<16} {:<9} {:<9} {:<9}'.format('Sign', 'Precision', 'Recall', 'F-Score'))
    print('-' * 46)
    for s in scores:
        print(s)
    print('-' * 46)
    print('{:<16} {:<9.3f} {:<9.3f} {:<9.3f}'.format('Overall', *_prf(overall_tp, overall_fp, overall_fn)))
    print('=' * 46)
