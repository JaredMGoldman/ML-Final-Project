import pandas as pd
import numpy as np
import random
import re


def is_city(ru_cc):
    """
    helper function to determine is_city feature
    """
    if 12 % ru_cc == 0:
        return 1.0
    return 0 

def winner(dem, rep, other):
    """
    helper function to determine label
    """
    if dem >= rep and dem >= other:
        return 0
    else:
        return 1

def pct_calc(series):
    """
    helper function to turn numbers between 0 and 100 to percentages
    """
    if re.search('pct', series.name):
        return series / 1e2
    return series

df = pd.read_csv("./election-context-2018.csv")         # load dataset
df['cvap_pct'] = df['cvap']/df['total_population']      # create cvap_pct feature
df['is_city'] = df['ruralurban_cc'].apply(is_city)      # create is_city feature
df['log_population'] = np.log(df['total_population'])   # take log of total population
df = df.apply(pct_calc)                                 # make everything into a percentage
                                                        # normalize the rest of the unnormalized features
df['median_hh_inc'] = (df['median_hh_inc'] - df['median_hh_inc'].min()) / df['median_hh_inc'].max()
df['median_hh_inc'] = df['median_hh_inc'] / df['median_hh_inc'].max()
df['log_population'] = (df['log_population'] - df['log_population'].min()) 
df['log_population'] = df['log_population'] / df['log_population'].max()
df = df.drop('total_population', axis = 1)
df = df.drop(1798, axis = 0)                            # remove bad data
df = df.drop(2388, axis = 0)


std_keys = ['cvap_pct', 'log_population', 'white_pct', 'black_pct', 'hispanic_pct', 
        'foreignborn_pct', 'female_pct', 'age29andunder_pct', 
        'age65andolder_pct', 'median_hh_inc', 'clf_unemploy_pct',
        'lesshs_pct', 'lesscollege_pct', 'rural_pct', 'is_city']

pres16_results_keys = ['trump16', 'clinton16', 'otherpres16']
pres12_results_keys = ['romney12', 'obama12', 'otherpres12']
gov16_results_keys = ['demgov16', 'repgov16', 'othergov16']
gov14_results_keys = ['demgov14', 'repgov14', 'othergov14']
sen16_results_keys = ['demsen16', 'repsen16', 'othersen16']
house16_results_keys = ['demhouse16', 'rephouse16', 'otherhouse16']

filenames = ["pres16.pkl", "pres12.pkl", "sen16.pkl", 
             "house16.pkl", "gov16.pkl", "gov14.pkl"]

results_keys = [pres16_results_keys, pres12_results_keys, 
                gov16_results_keys, gov14_results_keys, 
                sen16_results_keys, house16_results_keys]

# create individual datasets

pres16 = df[std_keys + pres16_results_keys].copy()
pres12 = df[std_keys + pres12_results_keys].copy()
gov16 = df[std_keys + gov16_results_keys].copy()
gov14 = df[std_keys + gov14_results_keys].copy()
house16 = df[std_keys + house16_results_keys].copy()
sen16 = df[std_keys + sen16_results_keys].copy()

data_frames = [pres16, pres12, gov16, gov14, sen16, house16]

def format_data(frames = data_frames, keylists = results_keys):
    """
        1) Eliminate samples where there is no race in the particular category [(gov, sen, house) + yr]
        2) Eliminate samples for races where a candidate runs without a Democratic or Republican challenger
        3) Determine winner: either Democrat (-1) or Non-Democrat (1) for remaining samples
        4) Remove the actual number of votes for each candidate

        return all the updated dataframes
    """

    for i in range(len(frames)):
        data_frame = frames[i]
        keylist = keylists[i]
        print(keylist, i)
        data_frame = data_frame.drop(data_frame[data_frame[keylist[0]].isnull()].index.tolist(), inplace = False)
        data_frame = data_frame.drop(data_frame[data_frame[keylist[0]] == 0].index.tolist(), inplace = False)
        data_frame = data_frame.drop(data_frame[data_frame[keylist[1]] == 0].index.tolist(), inplace = False)
        data_frame = data_frame.dropna(axis = 0, how = 'any')
        
        data_frame['winner'] = np.array(list(map(lambda x, y, z: winner(x,y,z), data_frame[keylist[0]].array, 
            data_frame[keylist[1]].array, data_frame[keylist[2]].array)))
        data_frame = data_frame.drop([keylist[0], keylist[1], keylist[2]], axis = 1)
        frames[i] = data_frame
    return frames
    
def save(frames = data_frames, frame_names = filenames):
    """
    save all of the dataframes to "name.pkl" files
    """
    path = "./data/"
    for i in range(len(frame_names)):
        frame = data_frames[i]
        filename = frame_names[i] 
        frame.to_pickle(path + filename)

def load_data(filename):
    """
    args:
        filename - name of dataset file to load

    returns: X, y
        X - features of dataset in type numpy array with shape (N_samples, N_features)
        y - labels of dataset in type numpy array with shape (N_samples, 1)
    """
    df = pd.read_pickle("./data/" + filename)
    iterable = df.iterrows()
    dataset = []
    while True:
        try:
            row = next(iterable)[1]
            label = row.pop('winner')
            dataset.append([np.array(row.array),label])
        except Exception:
            break
    random.shuffle(dataset)

    X = []
    y = []
    for feature, label in dataset:
        X.append(feature)
        y.append(label)

    return np.asarray(X), np.array(y)