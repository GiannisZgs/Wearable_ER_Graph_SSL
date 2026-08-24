#%% Constants

"MAKE A SCRIPT THAT PRINTS DATA OVERVIEW (FROM 1st processing.py end and Analysis.py end)"

import pytz
import os
import pandas as pd
import numpy as np
import gc
from datetime import timedelta as td
import gc
from functools import reduce
import numpy as np
import pandas as pd
from typing import Dict, Callable, Union, Tuple, List, Optional, Iterable
from datetime import timedelta as td
from scipy import stats
#import ray
import warnings
import time
from utils import load, dump, log, summary
from feature_extraction import extract

DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z" 
DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = os.path.join(os.getcwd(),'data')
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')

PATH_INTERMEDIATE = os.path.join(os.getcwd(),'intermediate')

data_subset = 'ScheduledResponseValid' 
#Options: ScheduledResponseValid, ScheduledResponseExcluded, VoluntaryResponse, ScheduledResponse

use_data = 'phone'
#Options: sensors, phone, all
label_extraction = 'all' 
#Options: all, one-by-one
want_label_features = True


DATA_TYPES = {
    'Acceleration': 'ACC',
    'AmbientLight': 'AML',
    'Calorie': 'CAL',
    'Distance': 'DST',
    'EDA': 'EDA',
    'HR': 'HRT',
    'RRI': 'RRI',
    'SkinTemperature': 'SKT',
    'StepCount': 'STP',
    'UltraViolet': 'ULV',
    'ActivityEvent': 'ACE',
    'ActivityTransition': 'ACT',
    'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT',
    'CallEvent': 'CAE',
    'Connectivity': 'CON',
    'DataTraffic': 'DAT',
    'InstalledApp': 'INS',
    'Location': 'LOC',
    'MediaEvent': 'MED',
    'MessageEvent': 'MSG',
    'WiFi': 'WIF',
    'ScreenEvent': 'SCR',
    'RingerModeEvent': 'RNG',
    'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS',
    'OnOffEvent': 'ONF'
}

SENSOR_TYPES  = {
    'Acceleration': 'ACC',
    'AmbientLight': 'AML',
    'Calorie': 'CAL',
    'Distance': 'DST',
    'EDA': 'EDA',
    'HR': 'HRT',
    'RRI': 'RRI',
    'SkinTemperature': 'SKT',
    'StepCount': 'STP',
    'UltraViolet': 'ULV'
}

PHONE_TYPES = {
    'ActivityEvent': 'ACE',
    'ActivityTransition': 'ACT',
    'AppUsageEvent': 'APP',
    'BatteryEvent': 'BAT',
    'CallEvent': 'CAE',
    'Connectivity': 'CON',
    'DataTraffic': 'DAT',
    'InstalledApp': 'INS',
    'Location': 'LOC',
    'MediaEvent': 'MED',
    'MessageEvent': 'MSG',
    'WiFi': 'WIF',
    'ScreenEvent': 'SCR',
    'RingerModeEvent': 'RNG',
    'ChargeEvent': 'CHG',
    'PowerSaveEvent': 'PWS',
    'OnOffEvent': 'ONF'
}

"Load participant demographics"
PARTICIPANTS = pd.read_csv(PATH_PARTICIPANT).set_index('pcode').assign(
    particpationStartDateTime=lambda x: pd.to_datetime(
        x['participationStartDate'], format='%Y-%m-%d'
    ).dt.tz_localize(DEFAULT_TZ)
)

"Load Processed Labels (computed from first_preprocessing.py)"

#LABELS_SCH_VALID_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseValid.csv'))

#LABELS_SCH_EXCL_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseExcluded.csv'))

#LABELS_VOL_PROC.to_csv(os.path.join(PATH_DATA, 'VoluntaryResponse.csv'))

#LABELS_SCHED_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponse.csv'))

LABELS_PROC = pd.read_csv(os.path.join(PATH_DATA,data_subset+'.csv'))
if data_subset == 'VoluntaryResponse':
    LABELS_PROC = LABELS_PROC.set_index(['pcode', 'responseTime'])
else:
    LABELS_PROC = LABELS_PROC.set_index(['pcode', 'timestamp'])


# # Feature Extraction

LABEL_VALUES = {'valence_bin':[1, 0],
                'arousal_bin':[1, 0]                
}
#'emo_quad':[0, 1, 2, 3]

WINDOW_DATA = {
    'S30': 30,
    'M01': 60,
    'M05': 60 * 5,
    'M10': 60 * 10,
    'M30': 60 * 30,
    'H01': 60 * 60,
    'H03': 60 * 60 * 3,
    'H06': 60 * 60 * 6
}

WINDOW_LABEL = {
    'H06': 60 * 60 * 6,
    'H12': 60 * 60 * 12,
    'H24': 60 * 60 * 24,
}

CATEGORIES = {
    'DST_MOT': ['IDLE', 'WALKING', 'JOGGING', 'RUNNING'],
    'ULV_INT': ['NONE', 'LOW', 'MEDIUM', 'HIGH'],
    'ACT': ['WALKING', 'STILL', 'IN_VEHICLE', 'ON_BICYCLE', 'RUNNING'],
    'APP_PAC': [],
    'APP_CAT': ['PERSONALIZATION', 'COMMUNICATION', 'PHOTOGRAPHY', 'FINANCE',
                'TOOLS', 'PRODUCTIVITY', 'HEALTH_AND_FITNESS', 'MISC',
                'VIDEO_PLAYERS', 'TRAVEL_AND_LOCAL', 'MAPS_AND_NAVIGATION',
                'LIFESTYLE', 'SYSTEM', 'MUSIC_AND_AUDIO', 'HOUSE_AND_HOME',
                'SOCIAL', 'GAME', 'SHOPPING', 'WEATHER', 'FOOD_AND_DRINK',
                'EDUCATION', 'NEWS_AND_MAGAZINES', 'ENTERTAINMENT', 'SPORTS',
                'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'LIBRARIES_AND_DEMO',
                'BEAUTY', 'ART_AND_DESIGN', 'AUTO_AND_VEHICLES'],
    'BAT_STA': ['CHARGING', 'DISCHARGING', 'FULL', 'NOT_CHARGING'],
    'CAE': ['CALL', 'IDLE'],
    'CON': ['DISCONNECTED', 'WIFI', 'MOBILE'],
    'LOC_CLS': [],
    'SCR': ['ON', 'OFF', 'UNLOCK'],
    'RNG': ['VIBRATE', 'SILENT', 'NORMAL'],
    'CHG': ['DISCONNECTED', 'CONNECTED'],
    'PWS': ['ACTIVATE', 'DEACTIVATE'],
    'ONF': ['ON', 'OFF']
}

RESAMPLE_S = {
    'ACC_AXX': 0.25,
    'ACC_AXY': 0.25,
    'ACC_AXZ': 0.25,
    'ACC_MAG': 0.25,
    'AML': 1.0,
    'EDA': 0.5, #was 0.5
}
#If not appearing here, all other modalities are resampled to 1Hz
"""FILT_FREQS = {
    'ACC_AXX': 1.8,
    'ACC_AXY': 1.8,
    'ACC_AXZ': 1.8,
    'ACC_MAG': 1.8,
    'AML': 0.5,
    'EDA': 0.5, #was 0.5
}"""

PINFO = PARTICIPANTS.assign(
    AGE=lambda x: x['age'],
    GEN=lambda x: x['gender'],
    BFI_OPN=lambda x: x['openness'],
    BFI_CON=lambda x: x['conscientiousness'],
    BFI_NEU=lambda x: x['neuroticism'],
    BFI_EXT=lambda x: x['extraversion'],
    BFI_AGR=lambda x: x['agreeableness'],
    PSS=lambda x: x['PSS'],
    PHQ=lambda x: x['PHQ'],
    GHQ=lambda x: x['GHQ'],
)[[
    'AGE', 'GEN', 'BFI_OPN', 'BFI_CON', 'BFI_NEU', 'BFI_EXT', 'BFI_AGR', 'PSS', 'PHQ', 'GHQ'
]]

PINFO = pd.get_dummies(PINFO, prefix_sep='=', dtype=bool).to_dict('index')
PINFO = {k: {f'PIF#{x}': y for x, y in v.items()} for k, v in PINFO.items()}

DATA = load(os.path.join(PATH_INTERMEDIATE, use_data + '_proc.pkl'))


# In[138]:
if label_extraction == 'one-by-one':

    #with on_ray(num_cpus=12):
    for l in ['valence', 'arousal']:#,'emo']:#, 'stress', 'disturbance']:
        
        labels = LABELS_PROC[f'{l}_bin'] if l != 'emo' else LABELS_PROC[f'{l}_quad']
        pids = labels.index.get_level_values('pcode').unique()

        feat = extract(
            pids=pids, 
            data=DATA,
            label=labels,
            label_values=LABEL_VALUES,
            window_data=WINDOW_DATA,
            window_label=WINDOW_LABEL,
            categories=CATEGORIES,
            constat_features=PINFO,
            resample_s=RESAMPLE_S,
            label_based_features=want_label_features,
            with_ray=False
        )

        dump(feat, os.path.join(PATH_INTERMEDIATE, use_data + '_' + data_subset +f'{l}.pkl'))

elif label_extraction == 'all':
    tags = ['valence_bin', 'arousal_bin', 'emo_quad']#,'stress', 'disturbance']
    labels = LABELS_PROC[tags]
    pids = labels.index.get_level_values('pcode').unique()

    feat = extract(
        pids=pids, 
        data=DATA,
        label=labels,
        label_values=LABEL_VALUES,
        window_data=WINDOW_DATA,
        window_label=WINDOW_LABEL,
        categories=CATEGORIES,
        constat_features=PINFO,
        resample_s=RESAMPLE_S,
        label_based_features=want_label_features,
        with_ray=False
    )

    dump(feat, os.path.join(PATH_INTERMEDIATE, use_data + '_' + data_subset +'_all_labels.pkl'))

# In[139]:
tags = ['valence_bin', 'arousal_bin']#, 'emo_quad']#,'stress', 'disturbance']
#for l in ['valence', 'arousal', 'stress', 'disturbance']:
X, y, group, t, stamps = load(os.path.join(PATH_INTERMEDIATE, use_data + '_' + data_subset +'_all_labels.pkl'))

print(f'- Feature space: {len(X.dtypes)}; Cat.: {np.sum(X.dtypes == bool)}; Num.: {np.sum(X.dtypes != bool)}')
for i,l in enumerate(tags):
    print(f'- Label distribution - {l}: {np.unique(y[:,i], return_counts=True)}')
"""
from datetime import datetime
#sumDifs = 0
difs = {p: [] for p in np.unique(group)}
allPsdiffs = []
for j,p in enumerate(np.unique(group)):
    stampsP = [stamps[s] for s in range(len(group)) if group[s] == p]
    for i in range(len(stampsP)-1):
        time0 = datetime.strptime(stamps[i], DATE_FORMAT)
        time1 = datetime.strptime(stamps[i+1], DATE_FORMAT)
        timediff = int((time1-time0).total_seconds())
        if timediff < 0:
            continue
        difs[p].append(timediff)
        allPsdiffs.append(timediff)
        #sumDifs += timediff

    print('#'*20 + '-----' +'#'*20)
    print(f'Participant {p}')
    print(f'Average time between samples: {np.mean(difs[p])}')
    print(f'Standard deviation of time between samples: {np.std(difs[p])}')
    print(f'Min time between samples: {min(difs[p])}')
    print(f'Max time between samples: {max(difs[p])}')

allPsdiffs = np.array(allPsdiffs)
import matplotlib.pyplot as plt
plt.hist(allPsdiffs[allPsdiffs<10000], bins=100, edgecolor='black')
plt.title(f'Histogram of time differences for all participants')
plt.xlabel('Time difference (seconds)')
plt.ylabel('Frequency')
plt.show()
"""
# Let's check whether the number of features is same as intented.

# In[140]:

N_NUM, N_CAT_B, N_CAT_NB = 0, 0, 0 

for k, v in DATA.items():
    if k in CATEGORIES:
        if CATEGORIES.get(k):
            N_CAT_B = N_CAT_B + 1
        else:
            N_CAT_NB = N_CAT_NB + 1
    else:
        N_NUM = N_NUM + 1

# Features relavant to delivery time
N_TIM = 7 + 2 + 7
print(f'N_TIM: {N_TIM}')

# Features relavant to personal demographics
N_PIF = len(PINFO['P01'])
print(f'N_PIF: {N_PIF}')
        
# Features relevant to latest value
N_VAL_NUM = N_NUM
N_VAL_CAT = sum([len(c)for c in CATEGORIES.values()])
N_VAL = N_VAL_NUM + N_VAL_CAT
print(f'N_VAL: {N_VAL} (N_VAL_NUM: {N_VAL_NUM} / N_VAL_CAT: {N_VAL_CAT})')

# Features relevant to duration since change
N_DSC = N_CAT_B + N_CAT_NB + sum([
    len(CATEGORIES.get(k))
    for k in CATEGORIES
])
print(f'N_DSC: {N_DSC}')


# Features from time-windows
N_WIN_NUM = N_NUM * 8 * len(WINDOW_DATA)
N_WIN_CAT = (N_CAT_B + N_CAT_NB) * 2 * len(WINDOW_DATA) + sum([
    len(WINDOW_DATA) if len(CATEGORIES.get(k)) == 2 else len(CATEGORIES.get(k)) * len(WINDOW_DATA)
    for k in CATEGORIES
])

print(f'N_WIN_NUM: {N_WIN_NUM} / N_WIN_CAT: {N_WIN_CAT}')


# Features from previous labels
N_LBL = len(WINDOW_LABEL) * (1 if len(LABEL_VALUES) <= 2 else len(LABEL_VALUES))
print(f'N_LBL: {N_LBL}')

N_FEAT = N_TIM + N_PIF + N_VAL + N_DSC + N_WIN_NUM + N_WIN_CAT + N_LBL
print(f'N_FEAT: {N_FEAT}')


# Okay, features are extracted as intended.


# %%
