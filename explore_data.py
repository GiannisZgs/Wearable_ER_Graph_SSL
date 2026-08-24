#%% Constants
from utils import load, dump, log, summary
from processing_utils import _load_data
import pytz
import os
import pandas as pd
import numpy as np
import gc
from datetime import timedelta as td

DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = os.path.join(os.getcwd(),'data')
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')

PATH_INTERMEDIATE = './intermediate'

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

"Dataset Overview"

import pandas as pd
import os


PARTICIPANTS = pd.read_csv(PATH_PARTICIPANT).set_index('pcode').assign(
    particpationStartDateTime=lambda x: pd.to_datetime(
        x['participationStartDate'], format='%Y-%m-%d'
    ).dt.tz_localize(DEFAULT_TZ)
)
#PARTICIPANTS.head()

"Participant summary"
#for c in PARTICIPANTS.columns:
#    print(f'- {c}:', summary(PARTICIPANTS[c]))


# ## Labels (via ESM)

#"Label data"

LABELS = pd.read_csv(PATH_ESM).set_index(
    ['pcode']
)
#LABELS.head()

"Label summary"

#for c in LABELS.columns:
#    print(f'- {c}:', summary(LABELS[c]))


"Group by participant"
inst = LABELS.groupby('pcode').count().iloc[:, -1]
"Obtain the number of responses that were given inside the scheduled time per participant"
inst_sch = LABELS.loc[lambda x: ~x['scheduledTime'].isna(), :].groupby('pcode').count().iloc[:, -1]
"Obtain the number of responses that were given voluntarily, i.e., outside the scheduled time per participant"
inst_vol = LABELS.loc[lambda x: x['scheduledTime'].isna(), :].groupby('pcode').count().iloc[:, -1]
"Create a new column 'timestamp' that contains the response time in datetime format"
"Assign the result to a new DF 'resp_time'"
resp_time = LABELS.assign(
    timestamp=lambda x: pd.to_datetime(x['responseTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ)
)
"For each subject, calculate time difference in seconds between consecutive responses"
sam = np.concatenate([
    (resp_time.loc[p, 'timestamp'].array - resp_time.loc[p, 'timestamp'].array.shift(1)).dropna().total_seconds()
    for p in LABELS.index.unique()
])

#print('- # Inst.:', summary(inst))
#print('- # Inst. - Scheduled:', summary(inst_sch))
#print('- # Inst. - Voluntary:', summary(inst_vol))
#print('- Samp. period:', summary(sam))
#for c in LABELS.columns:
#    print(f'- {c}:', summary(LABELS[c]))


"Load Sensor Data"
#%% 
STATS = {d: None for d in DATA_TYPES}


for data_type in DATA_TYPES:
    dat = _load_data(data_type,PATH_SENSOR)
    inst = dat.groupby('pcode').count().iloc[:, -1] #count instances of each group
    samp = np.concatenate([
        (dat.loc[(p,), :].index.array - dat.loc[(p,), :].index.array.shift(1)).dropna().total_seconds()
        for p in dat.index.get_level_values('pcode').unique()
    ])
    inst, samp = summary(inst), summary(samp)
    results = {'inst': inst, 'samp': samp}

    print('#'*5, data_type, '#'*5)
    print('- # Inst.:', inst)
    print('- Samp. period:', samp)


    # Append the results to the DataFrame
    STATS[data_type] = results
    

    for c in dat.columns:
        print(f'- {c}:', summary(dat[c]))
        
    del dat
    gc.collect()
    
STATS = pd.DataFrame(STATS)
