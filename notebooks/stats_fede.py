import numpy as np
import pandas as pd

############################################################################
########## THESE ARE THE FUNCTIONS FOR METRICS FROM F's THESIS #############
############################################################################

## Functions that are currently used in notebooks

def binaryResult(a): #1 if at least one ciclon 
    if a > 0:
      return 1
    else:
      return 0
    
def alertDelay(y,yhat):
  sum=0
  count=0
  #intervals=[]
  for i,x in enumerate(y):
    
    if x==1: 
      dif = np.argmax(yhat[i:])
      sum=sum+dif
      count=count+1
      #intervals.append(dif)
    
  #print('Delay intervals:',intervals)
  return sum/count

def alertDelay3(t, reals, pred):
  k = t*2-1

  reallist = [reals[i : min(i+k, len(reals))] for i in range(len(reals))]
  #display(*zip(pred, reallist))
  tfpositive = [computeTrueFalsePositive(x, xl, t) for x, xl in zip(pred,reallist) if x == 1]

  predlist =   [pred[max(i-k+1, 0): min(i+1, len(pred))] for i in range(len(reals))]
  #display(*zip(reals, predlist))
  tfnegative = [computeTrueFalseNegative(x, xl, t) for x, xl in zip(reals, predlist) if x == 1]

  return (tfpositive, tfnegative)


#########################################################
#########################################################
######### STATS FUNCTIONS TO EVALUATE MODELS ############

def computeTrueFalsePositive(y: int, l: list, t: int):
  
  assert(y == 1)

  for k in range(t):
    
    s = t-1
    if l[min(s+k, len(l)-1)] == 1 or l[max(s-k, 0)] == 1:
      return k
  
  return -1

def computeTrueFalseNegative(y: int, l: list, t: int):
  assert(y == 1)
  return max(l)

def falsePositive(df, s1, s2):
    df = pd.DataFrame(np.column_stack([s1, s2]), columns = ["y", "yhat"])
    df["cums"] = df["y"].cumsum()
    res = df[df['cums']!=0].groupby("cums")["yhat"].sum()
    res = [r - 1 if r != 0 else r for r in res]
    return sum(res)

### Function to compute all the stats to evaluate the model
### FP, TP, FN, 
def computeStats(t, reals, pred):
  pos, neg = alertDelay3(t, reals, pred)
  
  pos = np.array(pos)
  neg = np.array(neg)

  false_positive = np.count_nonzero(pos == -1)
  true_positive = np.count_nonzero(pos != -1)
  tot_distance = pos[pos!=-1].sum()
  avg_distance = tot_distance/true_positive
  false_negative = np.count_nonzero(neg == 0)

  std = np.std(pos[(pos!=-1)])
  
  return (false_positive, true_positive, false_negative, avg_distance, std)

###################################################
###################################################
############ TROPICAL CYCLONES STATS ##############
def computeTCStats(target, predictions, steps, offset, prediction_len):
  d = []

  for i in range(steps):
    stats = computeStats(i+1, target[-prediction_len-i:],predictions[i + offset])

    # Tot cyclones
    tot_cyclones = target[-prediction_len:].count(1)
    # Number of real anomalies
    real_anomalies = target[-prediction_len:].count(1)
    # Number of predicted anomalies
    predicted_anomalies = np.count_nonzero(predictions[i + offset] == 1)
    # FP
    fp = stats[0]
    # TP 
    tp = stats[1] # TODO: I think there is an error in the computation of true positive class
    # FN
    fn = stats[2]
    # TN
    tn = (np.count_nonzero(predictions[i + offset] == 0)) - fn
    # MEAN
    mean = stats[3]

    # FP RATE
    fp_rate = "{:10.2f}%".format(fp/predicted_anomalies*100)
    # TP RATE - PRECISION
    precision = "{:10.2f}%".format(tp/predicted_anomalies*100)
    # TN RATE
    tn_rate = "{:10.2f}%".format(tn/(np.count_nonzero(predictions[i + offset] == 0))*100)
    # FN RATE
    fn_rate = "{:10.2f}%".format(fn/np.count_nonzero(predictions[i + offset] == 0)*100)
    # STD
    std = stats[4]
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
    accuracy = ( tp + tn ) / prediction_len

    d.append([
      "t+{}".format(i+1), # Time index
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
      ])
    
  return d