import numpy as np

# This function shift the dataset of n steps,
# It is important when we want to make a prediction for a possible cyclon
# in the next n-days
def shift_dataset(dataset, steps):
    new_dataset = []

    new_dataset = [dataset[steps:]]
    for i in range(steps):
        new_dataset.append(dataset[(steps-1)-i : -i-1])
    return new_dataset



# This function compute the new target array considering a time window
# For each n-steps if there is at least one positive classified sample,
# then that target is considered as positive
# Only if in the n-steps all the values are negative, we'll have a negative class

# We're considering to predict if in the following n-days there will be a cyclone,
# so if in the n days there is one cyclon, we have to predict a positive result. 
def compute_target_window(target, steps, max_steps):
    assert steps <= max_steps    
    new_target = []
    
    if steps > 0:
        for i in range(max_steps-steps, len(target)-steps):
            if target[(max_steps-steps+i) : (max_steps+i)].value_counts()[1] > 0:
                new_target.append(1)
            else:
                new_target.append(0)
    else:
        new_target = target[max_steps:].tolist()
        
    return new_target

# This function returns a list of N arrays,
# Each array a version of the original target_list, recomputed considering the 
# presence of a TC in the next n-days (steps)
def compute_NSteps_target(target_list, steps):
    new_target_list = []
    for i in range(steps):
        new_target_list.append(compute_target_window(target_list, i, 9))
    return new_target_list



# This function constructs a new dataset, receiving as parameter the old one - set.
# Each new sample is taken from a window of a fixed size
# If exists at least one point that is equal to one in this window,
# the new datapoint is a positive clas.
# Otherwise (only if all the samples in the window are negative classes)
# it will the a negative class.

# I'm considering part of the window all the following t-1 samples
# For example if we consider a sample at timestep t, the window is composed
# of all samples from t to t+window_size -> window[t : t + window_size]

# In this case the window size refers to the length of the window
# Having window_size = 0 (impossible) or window_size = 1 corresponds to have
# a window that contains only 1 sample
def reshape_window_future(X_set, Y_set, window_size): 
    if window_size > 1:
        new_set_size = len(Y_set) - window_size + 1
        new_X_set = np.zeros(shape=(new_set_size, X_set.shape[1]))
        new_Y_set = np.zeros(new_set_size)   
        
        new_X_set = X_set[0 : new_set_size]
        
        for i in range(new_set_size):
            window = Y_set[i : i + window_size]
            # print("===============================")
            # print("Pivot idx: ", i)
            # print("Window: ", window)
            if (1) in window:
                new_Y_set[i] = 1
                # print("New target: ", 1)
            else:
                new_Y_set[i] = 0
                # print("New target: ", 0)
        return new_X_set, new_Y_set
    else:
        return X_set, Y_set
    
    
# This function is very similar to the reshape_window_future, but the
# window is composed of both past and positive elements.
# In particular, if we consider the sample of set at timestep t, 
# the window is composed of all the elements from (t - steps) to (t + steps), where steps
# is the number of the maximum distance from the initial sample

# For example: set = [1, 1, 0, 0, 0, 1, 0, 1, 0]
# considering the sample at timestep = 4 (that is set[3] = 0 - and it is the pivot), the correspondent
# window with steps = 2 will be: window = [1, 0, 0, 0, 1] 

def reshape_window_neighbourhood(X_set, Y_set, steps): # Time window have size = (2*steps + 1)
    if steps > 0:
        new_set_size = len(Y_set) - steps*2
        new_X_set = np.zeros(shape=(new_set_size, X_set.shape[1]))
        new_Y_set = np.zeros(new_set_size) 
        
        # Reshape X_set to have the same number of samples of new_Y_set
        new_X_set = X_set[steps : (len(Y_set) - steps)]
    
        # Compute the new Y_set
        new_Y_set_index = 0
        for i in range(steps, len(Y_set) - steps):
            window = Y_set[i-steps : i+steps+1]
            # print("===============================")
            # print("Pivot idx: ", i)
            # print("Window: ", window)
            if (1) in window:
                new_Y_set[new_Y_set_index] = 1
                # print("New target: ", 1)
            else:
                new_Y_set[new_Y_set_index] = 0
                # print("New target: ", 0)
            new_Y_set_index = new_Y_set_index + 1                
        return new_X_set, new_Y_set
    else:
        return X_set, Y_set