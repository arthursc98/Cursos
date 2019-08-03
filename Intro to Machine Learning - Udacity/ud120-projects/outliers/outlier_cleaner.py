#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    error=[a-b for a,b in zip(predictions,net_worths)]
    full_data=zip(ages,net_worths,error)
    sorted_data=sorted(full_data,key= lambda x:x[2],reverse=True)
    cleaned_data=sorted_data[int(len(sorted_data)*0.1):]
    
    return cleaned_data

