import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    assert len(prediction) == len(ground_truth)
    num_samples = len(prediction)
    
    tp = np.sum([ground_truth[i] == prediction[i] == True for i in range(num_samples)])
    fp = np.sum([True for i in range(num_samples) \
                if ground_truth[i] == False and prediction[i] == True])
    fn = np.sum([True for i in range(num_samples) \
                if ground_truth[i] == True and prediction[i] == False])
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = np.sum(prediction == ground_truth) / num_samples
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = 0
    num_samples = len(prediction)
    accuracy = np.sum(prediction == ground_truth) / num_samples    
    
    return accuracy
