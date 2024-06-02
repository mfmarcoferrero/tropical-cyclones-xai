###############################################################################
###############################################################################
######################## FUNCTIONS TO MANAGE DATES ############################

from datetime import datetime
from datetime import timedelta

def dates_list(str_start, str_end):
    start_dt = datetime.strptime(str_start, "%Y/%m/%d").date()
    end_dt = datetime.strptime(str_end, "%Y/%m/%d").date()

    # difference between current and previous date
    delta = timedelta(days=1)

    # store the dates between two dates in a list
    dates = []

    while start_dt <= end_dt:
        # add current date to list by converting  it to iso format
        dates.append(start_dt)
        # increment start date by timedelta
        start_dt += delta
    return(dates)

# str_d1 and str_d2 must be formatted as YYYY/MM/DD
def date_diff(str_d1, str_d2):
    # convert string to date object
    d1 = datetime.strptime(str_d1, "%Y/%m/%d").date()
    d2 = datetime.strptime(str_d2, "%Y/%m/%d").date()

    # difference between dates in timedelta
    delta = d2 - d1
    return delta

###############################################################################
###############################################################################
#################### FUNCTIONS TO SELECT PART OF A DF #########################

def select_one_year(df_input, year_str):
    str_start = '{}/01/01'.format(year_str)
    str_end =  '{}/12/31'.format(year_str)

    start_dt = datetime.strptime(str_start, "%Y/%m/%d").date()
    end_dt = datetime.strptime(str_end, "%Y/%m/%d").date()

    df_year = df_input[df_input['Date'] >= start_dt]
    df_year = df_year[df_year['Date'] <= end_dt]
    return df_year

def select_multiple_years(df_input, start_year_str, end_year_str):
    str_start = '{}/01/01'.format(start_year_str)
    str_end =  '{}/12/31'.format(end_year_str)

    start_dt = datetime.strptime(str_start, "%Y/%m/%d").date()
    end_dt = datetime.strptime(str_end, "%Y/%m/%d").date()

    df_year = df_input[df_input['Date'] >= start_dt]
    df_year = df_year[df_year['Date'] <= end_dt]
    return df_year


###############################################################################
###############################################################################
#################### FUNCTIONS TO BUILD NEW FEATURES ##########################
import numpy as np
from order_of_magnitude import order_of_magnitude

# Label a dataset with "increasing" and "decreasing" if data has growing tendency
# Using this function and evaluating the direction of gradient 
# we can establish if a point is in increasing or decreasing tendency
def increasing_decreasing_data(data):
    # Load and preprocess your dataset, assuming it's stored in a variable named 'data'
    # Calculate derivatives
    derivatives = np.gradient(data)

    # Initialize an empty list to store labels
    labels = []

    # Iterate through the data points to label them
    for i in range(len(data)):
        if derivatives[i] > 0:
            labels.append('increasing')
        elif derivatives[i] < 0:
            labels.append('decreasing')
        else:
            labels.append('unchanged')

    # Now, 'labels' contains the classification of each data point
    return labels


# Label a dataset with "max" or "min" if the current point
# is a local maximum or local minimum of the curve.
# Each point considered max or min only if exceeds the sum of the mean value and the standard deviation

# Considering that the np.gradient function doesn't return values that are exaclty zero, a tolerance
# is fixed to consider lowest order of magnitude allowed to classify a point as a local min or max (e.g. 1e-3)
def find_local_max_min(data, tolerance, avg_value, std_value):
    # Load and preprocess your dataset, assuming it's stored in a variable named 'data'
    # Calculate derivatives
    derivatives = np.gradient(data)
    # Initialize an empty list to store labels
    local_max = []
    local_min = []

    # Iterate through the data points to label them
    for i in range(0, len(data)):
        ord_mag = order_of_magnitude.order_of_magnitude(abs(derivatives[i]))
        if ord_mag < tolerance and i > 0 and i < len(data) - 1: # Point is considered to be zero
            # # Check for local maxima and minima
            if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > avg_value + std_value: ## Local Max coditions
                local_max.append("max")
                local_min.append(None)
            elif data[i] < data[i - 1] and data[i] < data[i + 1] and data[i] < avg_value - std_value: ## Local Min coditions
                local_max.append(None)
                local_min.append("min")
            else:
                local_max.append(None)
                local_min.append(None)
        else:
            local_max.append(None)
            local_min.append(None)
    
    # Now, 'labels' contains the classification of each data point
    return np.array(local_max), np.array(local_min)



# Build the neighbourhood of a local max-min considering a window of fixed size 
# (window_size = 2*steps) as parameter of the function
def build_neighborhood_max_min(data, steps):
    window_size = steps*2
    neighbourhood_data = np.full(data.shape, None)
    i = 0
    while i < len(data):
        if data[i] != None and i - steps > 0 and i + steps < len(data):
            neighbourhood_data[i-steps : i+steps+1] = np.full(window_size+1, data[i])
            i = i + steps
        else:
            #neighbourhood_data.append(None)
            i = i + 1
    return neighbourhood_data