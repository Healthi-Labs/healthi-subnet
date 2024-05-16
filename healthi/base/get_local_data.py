import numpy as np
import pandas as pd
from datetime import datetime


def get_local_data(hotkey):
    """
    This function serves the function of getting data when the validator
    can not acquire the data from the data API.
    """

    # Read data and Set weight
    DATA = pd.read_parquet('./healthi/base/data/trainset.parquet').rename(columns={'code': 'EHR','time': "admission_time"}, inplace=False)
    sampled_index = np.random.randint(low=0, high=len(DATA), size=1, dtype=int)[0]
    sampled_data = DATA.iloc[sampled_index]
    label = np.array(sampled_data['label'])
    label_weight = (label * 0.95) + (1-label)*0.05

    # Create Keys and Values
    current_time = datetime.now()
    print_time = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
    sampled_data['created_at'] = print_time
    sampled_data['label_weight'] = label_weight
    sampled_data['weight'] = 0.01
    sampled_data['hotkey'] = hotkey
    sampled_data['task'] = 'Disease Prediction'
    return_data = sampled_data.to_json()

    return return_data

