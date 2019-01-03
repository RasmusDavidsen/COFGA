# This script calculates the MAP score

from sklearn.metrics import precision_score
import numpy as np

def mAP_score(predicted, validation):
    classes = predicted.shape[1]
    pred_bin = np.zeros(classes)
    threshold = 0.5

    # binarizing output matrix
    predicted_bin = (predicted > threshold).astype(np.int_)

    AP_class = np.zeros(classes)

    for j in range(classes):
        temp_predicted = predicted_bin[:,j]

        temp_validation = validation[:,j]

        AP_class[j] = precision_score(temp_validation, temp_predicted, average="binary")


    mAP = np.average(AP_class)

    return (mAP, AP_class)
