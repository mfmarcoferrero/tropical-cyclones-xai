import numpy as np
import pandas as pd

########################################################
########################################################
######## STANDARD FUNCTIONS TO COMPUTE METRICS #########

# Model correctly predicts the positive class
def true_positive_1(predictions, targets):
    true_positives = 0
    
    # Iterate through each prediction and target pair
    for prediction, target in zip(predictions, targets):
        # Check if the prediction is positive and matches the target
        if prediction == 1 and target == 1:
            true_positives += 1
    
    return true_positives

# Model wrongly predicts the positive class
def false_positive_1(predictions, targets):
    false_positive = 0
    
    # Iterate through each prediction and target pair
    for prediction, target in zip(predictions, targets):
        if prediction == 1 and target == 0:
            false_positive += 1
    
    return false_positive

def true_negative_1(predictions, targets):
    true_negative = 0
        
    for prediction, target in zip(predictions, targets):
        if prediction == 0 and target == 0:
            true_negative += 1
        
    return true_negative

def false_negative_1(predictions, targets):
    false_negative = 0
    
    for prediction, target in zip(predictions, targets):
        if prediction == 1 and target == 0:
            false_negative += 1
    
    return false_negative

def compute_mean(predictions, targets):

  return

def compute_std(predictions, targets):

    return

# Take as inputs two LISTS target and predictions
def TCStats(targets, predictions, timestep):
  assert len(targets) == len(predictions)

  d = []

  N = len(predictions)

  # Tot cyclones
  tot_cyclones = targets.count(1)
  # Number of real anomalies
  real_anomalies = targets.count(1)
  # Number of predicted anomalies
  predicted_anomalies = predictions.count(1)
  negative_predictions = predictions.count(0)
  
  # FP
  fp = false_positive_1(predictions, targets)
  # TP 
  tp = true_positive_1(predictions, targets) # TODO: There is an error in the computation of true positive class
  # FN
  fn = false_negative_1(predictions, targets)
  # TN
  tn = true_negative_1(predictions, targets)
 
  # MEAN
  mean = compute_mean(predictions, targets)
  # STD
  std = compute_std(predictions, targets)

  # FP RATE
  fp_rate = "{:10.2f}%".format(fp/predicted_anomalies*100)
  # TN RATE
  tn_rate = "{:10.2f}%".format(tn/negative_predictions*100)
  # FN RATE
  fn_rate = "{:10.2f}%".format(fn/negative_predictions*100)
  # TP RATE - PRECISION
  precision = "{:10.2f}%".format(tp/predicted_anomalies*100)
  
  # RECALL
  recall = tp / (tp+fn)
  # F1 (2*precision*recall/precision+recall)
  p = tp/predicted_anomalies
  f1 = 2*(p*recall)/(p + recall)
  # FAR (False Allarm Ratio)
  far = fp / (fp + tp)
  # POD (Probability of Detection) = TP / TotCyclones
  pod = tp / tot_cyclones #POD = Tp/totCiclon
  # ACCURACY
  accuracy = ( tp + tn ) / N

  d = [
    "t+{}".format(timestep+1), # Time index
    real_anomalies,
    predicted_anomalies,
    mean,
    fp,
    tp,
    fn,
    tn,
    fp_rate,
    precision,
    tn_rate,
    fn_rate,
    std,
    recall,
    f1,
    far,
    pod,
    accuracy
    ]
  
  return d

def compute_10steps_stats(targets, predictions):
  d = []
  for i in range(10):
    d.append(TCStats(targets[i], predictions[i].tolist(), i))
  
  return d


#############################################################
#############################################################
#############################################################
######## METRICS RECONSIDERING THE WINDOW HORIZON ###########


# This function count the number of predictions that in the next n days are correct
def true_positive_horizon(predictions, targets, time_horizon):
    assert (len(predictions) == len(targets) + time_horizon)
    true_positives = 0
    isTP = False

    for i in range(len(predictions)):
        isTP = False
        if predictions[i] == 1:
            end = i + time_horizon if i + time_horizon < len(targets) else len(targets) -1
            if end != i:
                for j in range(i, end):
                    if targets[j] == 1:
                        isTP = True
            else:
                if targets[i] == 1:
                    isTP = True
            if isTP:
                true_positives = true_positives + 1
    return true_positives

def false_positive_horizon(predictions, targets, time_horizon):
    assert (len(predictions) == len(targets) + time_horizon)
    false_positives = 0
    isFP = True

    for i in range(len(predictions)):
        if predictions[i] == 1:
            isFP = True
            end = i + time_horizon if i + time_horizon < len(targets) else len(targets) -1
            for j in range(i, end):
                if targets[j] == 1:
                    isFP = False
            if isFP:
                false_positives = false_positives + 1
    return false_positives

def true_negative_horizon(predictions, targets, time_horizon):
    assert (len(predictions) == len(targets) + time_horizon)
    true_negatives = 0
    isTN = False

    for i in range(len(predictions)):
        if predictions[i] == 0:
            isTN = True
            end = i + time_horizon if i + time_horizon < len(targets) else len(targets) -1
            for j in range(i, end):
                if targets[j] == 1:
                    isTN = False
            if isTN:
                true_negatives = true_negatives + 1
    return true_negatives

def false_negative_horizon(predictions, targets, time_horizon):
    assert (len(predictions) == len(targets) + time_horizon)
    false_negatives = 0
    isFN = False

    for i in range(len(predictions)):
        if predictions[i] == 0:
            isFN = False
            end = i + time_horizon if i + time_horizon < len(targets) else len(targets) -1
            if (i != time_horizon):
                for j in range(i, end):
                    if targets[j] == 1:
                        isFN = True
            else:
                if targets[i] == 1:
                    isFN = True
            if isFN:
                false_negatives = false_negatives + 1
    return false_negatives
  
  
def TCStats_horizon(targets, predictions, time_horizon):
  print("Target len: ", len(targets))
  print("Prediction len: ", len(predictions))

  assert (len(predictions) + time_horizon == len(targets))

  d = []

  N = len(predictions)

  # Tot cyclones
  tot_cyclones = targets.count(1)
  # Number of real anomalies
  real_anomalies = targets.count(1)
  # Number of predicted anomalies
  predicted_anomalies = predictions.count(1)
  negative_predictions = predictions.count(0)

  # FP
  fp = false_positive_horizon(predictions, targets, time_horizon)
  # TP 
  tp = true_positive_horizon(predictions, targets, time_horizon) # TODO: There is an error in the computation of true positive class
  # FN
  fn = false_negative_horizon(predictions, targets, time_horizon)
  # TN
  tn = true_negative_horizon(predictions, targets, time_horizon)

  print("TP: ", tp)
  print("FP: ", fp)
  print("TN: ", tn)
  print("FN: ", fn)

  # MEAN
  mean = compute_mean(predictions, targets)
  # STD
  std = compute_std(predictions, targets)

  # FP RATE
  fp_rate = "{:10.2f}%".format(fp/predicted_anomalies*100)
  # TN RATE
  tn_rate = "{:10.2f}%".format(tn/negative_predictions*100)
  # FN RATE
  fn_rate = "{:10.2f}%".format(fn/negative_predictions*100)
  # TP RATE - PRECISION
  precision = "{:10.2f}%".format(tp/predicted_anomalies*100)

  # RECALL
  recall = tp / (tp+fn)
  # F1 (2*precision*recall/precision+recall)
  p = tp/predicted_anomalies
  f1 = 2*(p*recall)/(p + recall)
  # FAR (False Allarm Ratio)
  far = fp / (fp + tp)
  # POD (Probability of Detection) = TP / TotCyclones
  pod = tp / tot_cyclones #POD = Tp/totCiclon
  # ACCURACY
  accuracy = ( tp + tn ) / N

  d = [
    "t+{}".format(time_horizon+1), # Time index
    real_anomalies,
    predicted_anomalies,
    mean,
    fp,
    tp,
    fn,
    tn,
    fp_rate,
    precision,
    tn_rate,
    fn_rate,
    std,
    recall,
    f1,
    far,
    pod,
    accuracy
    ]

  return d

def compute10steps_horizon(targets, predictions):
    d = []
    for i in range(10):
      d.append(TCStats_horizon(targets[-len(predictions[i])+i:], predictions[i].tolist(), i))
    return d