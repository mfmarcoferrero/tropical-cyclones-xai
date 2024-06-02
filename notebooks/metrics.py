import numpy as np
import math
from tabulate import tabulate
from sklearn import metrics
import matplotlib as plt

# In this file we'lle use only np.arrays
########################################################
########################################################
######## STANDARD FUNCTIONS TO COMPUTE METRICS #########

# Model correctly predicts the positive class
def true_positive(predictions, targets):
    true_positives = 0
    # Iterate through each prediction and target pair
    for prediction, target in zip(predictions, targets):
        # Check if the prediction is positive and matches the target
        if prediction == 1 and target == 1:
            true_positives += 1
    return true_positives

# Model wrongly predicts the positive class
def false_positive(predictions, targets):
    false_positive = 0
    # Iterate through each prediction and target pair
    for prediction, target in zip(predictions, targets):
        if prediction == 1 and target == 0:
            false_positive += 1
    return false_positive

def true_negative(predictions, targets):
    true_negative = 0
    for prediction, target in zip(predictions, targets):
        if prediction == 0 and target == 0:
            true_negative += 1
    return true_negative

def false_negative(predictions, targets):
    false_negative = 0
    for prediction, target in zip(predictions, targets):
        if prediction == 0 and target == 1:
            false_negative += 1
    return false_negative

###################################################################
###################################################################
################ FUNCTIONS TO EVALUATE THE DITANCE ################
############# OF PREDICTION IN WINDOW OF FIXED SIZE ###############

# Compute the mean distance from the current timestep to the first positive prediction
# inside the time window, considering windows in the future
def compute_mean_future_window(predictions, window_targets, real_targets, window_size):
    assert len(predictions) == len(window_targets)
    assert (len(window_targets) + window_size - 1) == len(real_targets) 
    stop = False
    mean = 0
    distance_sum = 0
    tot_predictions = np.count_nonzero(predictions)
    if window_size > 1:
        for i in range(len(predictions)):
            if predictions[i] == 1:
                window = real_targets[i : i + window_size]
            stop = False
            for j in range(window_size):
                if window[j] == 1 and not stop:
                    distance_sum = distance_sum + j
                    stop = True
    mean = distance_sum/tot_predictions
    return mean

# Compute the standard deviation of the distance fron the current timestep to the first positive
# prediction in window (only in the future)
def compute_std_future_window(predictions, window_targets, real_targets, window_size, mean):
  assert len(predictions) == len(window_targets)
  assert (len(window_targets) + window_size - 1) == len(real_targets)
  stop = False
  std = 0
  sqr_distance_sum = 0
  tot_predictions = np.count_nonzero(predictions)

  if window_size > 1:
    for i in range(len(predictions)):
      if predictions[i] == 1:
        window = real_targets[i : i + window_size]
        stop = False
        for j in range(window_size):
          if window[j] == 1 and not stop:
            sqr_distance = (j - mean)**2
            sqr_distance_sum = sqr_distance_sum + sqr_distance
            stop = True
    std = math.sqrt(sqr_distance_sum/tot_predictions)
  return std

#########################################################################
#########################################################################
################## FUNCTIONS TO RETURN ALL THE METRICS ##################

# Take as inputs two LISTS target and predictions
def TCStats(targets, predictions, real_targets, timestep):
  assert len(targets) == len(predictions)
  d = []
  N = len(predictions)
  # Number of real anomalies
  real_anomalies = np.count_nonzero(targets)
  
  # Number of predicted anomalies
  positive_predictions = np.count_nonzero(predictions)
  negative_predictions = len(predictions) - positive_predictions
  # print(targets.shape, predictions.shape)
  # print(tot_cyclones)
  # print(predicted_anomalies)
  # print(negative_predictions)
  
  # FP
  fp = false_positive(predictions, targets)
  fp_rate = "{:10.2f}%".format(fp/positive_predictions*100) # FP RATE
  
  # TP 
  tp = true_positive(predictions, targets)
  tp_rate = "{:10.2f}%".format(tp/positive_predictions*100) # TP RATE - PRECISION 

  # FN
  fn = false_negative(predictions, targets)
  fn_rate = "{:10.2f}%".format(fn/negative_predictions*100)  # FN RATE

  # TN
  tn = true_negative(predictions, targets)
  tn_rate = "{:10.2f}%".format(tn/negative_predictions*100) # TN RATE

  # MEAN and STD
  window_size = timestep
  #mean = compute_mean_future_window(predictions, targets, real_targets, timestep)
  #std = compute_std_future_window(predictions, targets, real_targets, timestep, mean)
  mean = 0
  std = 0
  
  # RECALL
  recall = tp / (tp+fn)
  # F1 (2*precision*recall/precision+recall)
  p = tp/positive_predictions
  f1 = 2*(p*recall)/(p + recall)
  # FAR (False Allarm Ratio)
  far = fp / positive_predictions
  # POD (Probability of Detection) = TP / TotCyclones
  pod = tp / real_anomalies #POD = Tp/totCiclon
  # ACCURACY
  accuracy = ( tp + tn ) / N
  
  d = [
    # "t+{}".format(timestep), # Time index
    window_size,
    real_anomalies,
    positive_predictions,
    round(mean,4),
    round(std, 4),
    fp,
    tp,
    fn,
    tn,
    fp_rate,
    tp_rate,
    tn_rate,
    fn_rate,
    round(recall,4),
    round(f1,4),
    round(far,4),
    round(pod,4),
    round(accuracy,4)
    ]
  return d