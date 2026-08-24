#%% Constants
from utils import load, dump, log, summary
from processing_utils import _load_data
import processing_utils as proc
import pytz
import os
import pandas as pd
import numpy as np
import gc
from datetime import timedelta as td
import gc
from functools import reduce

DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

PATH_DATA = os.path.join(os.getcwd(),'data')
PATH_ESM = os.path.join(PATH_DATA, 'EsmResponse.csv')
PATH_PARTICIPANT = os.path.join(PATH_DATA, 'UserInfo.csv')
PATH_SENSOR = os.path.join(PATH_DATA, 'Sensor')

PATH_INTERMEDIATE = os.path.join(os.getcwd(),'intermediate')

use_data = 'all'
#Options: sensors, phone, all

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


PARTICIPANTS = pd.read_csv(PATH_PARTICIPANT).set_index('pcode').assign(
    particpationStartDateTime=lambda x: pd.to_datetime(
        x['participationStartDate'], format='%Y-%m-%d'
    ).dt.tz_localize(DEFAULT_TZ)
)
#PARTICIPANTS.head()

"Participant summary"
#for c in PARTICIPANTS.columns:
#    print(f'- {c}:', summary(PARTICIPANTS[c]))


"Label data"

LABELS = pd.read_csv(PATH_ESM).set_index(
    ['pcode']
)
#LABELS.head()


"Start Preprocessing"

# Here we consider binary classifications for valence, arousal, stress, and disturbance, in which a label value greater than 0 is "HIGH" (1) and the rest is "LOW" (0), at the arrival of ESM prompts (*scheduledTime*)

# ## Label
# 
# Because we intended to collected participants' responses to ESMs not voluntary responses, we screend out some responses as follows:
# * We first screen out ESM responses that does not have 'scheduledTime' (meaning that a given ESM was expired or participants voluntarily reported their affective states regardless of ESM delivery). 
# * Since we will evaluate our model using LOSO, the small number of responses for each participant might lead to inappropriate performance evaluation. We emprically set the number of the minimum responses upon ESM delivery as 5 per day (i.e., a half of our guides), so that we excluded participants whose responses to ESM less than 35.

"Isolate the three groups of labels: Scheduled valid, voluntary, and scheduled excluded"
"Isolation happens in the participant level, not individual ESM responses"

"Include all the below label processing in processing_utils.py"

LABELS_SCHED = LABELS.loc[
    lambda x: ~x['scheduledTime'].isna(), :
]
print(f'# Non-voluntary response: {len(LABELS_SCHED)}')
print(summary(LABELS_SCHED.groupby('pcode').count().iloc[:, -1]))
#LABELS_SCHED.to_csv(os.path.join(PATH_DATA, 'ScheduledResponse.csv'))

LABELS_VOL = LABELS.loc[
    lambda x: x['scheduledTime'].isna(), :                       
]
print(f'# Voluntary response: {len(LABELS_VOL)}')
print(summary(LABELS_VOL.groupby('pcode').count().iloc[:, -1]))
vol_pcode = LABELS_VOL.groupby('pcode').count().iloc[:, -1]
print(vol_pcode, f'#participants = {len(vol_pcode)} / #response = {sum(vol_pcode)}')
#LABELS_VOL.to_csv(os.path.join(PATH_DATA, 'VoluntaryResponse.csv'))


excl_pcode = LABELS_SCHED.loc[
    lambda x: ~x['scheduledTime'].isna(), :
].groupby('pcode').count().iloc[:, -1].loc[lambda y: y < 35]

incl_pcode = LABELS_SCHED.loc[
    lambda x: ~x['scheduledTime'].isna(), :
].groupby('pcode').count().iloc[:, -1].loc[lambda y: y >= 35]

LABELS_SCH_EXCL = LABELS_SCHED.loc[
    lambda x: x.index.get_level_values('pcode').isin(excl_pcode.index),:
]
#LABELS_SCH_EXCL.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseExcluded.csv'))

LABELS_SCH_VALID = LABELS_SCHED.loc[
    lambda x:  ~x.index.get_level_values('pcode').isin(excl_pcode.index), :
]
#LABELS_SCH_VALID.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseValid.csv'))

print(f'# Response from participants with enough responses: {len(LABELS_SCH_VALID)}')
print(incl_pcode, f'#participants = {len(incl_pcode)} / #response = {sum(incl_pcode)}')
print(summary(LABELS_SCH_VALID.groupby('pcode').count().iloc[:, -1]))
print('# Participants whose responses to ESM delivery were less then 35')
print(excl_pcode, f'#participants = {len(excl_pcode)} / #response = {sum(excl_pcode)}')

"Label Processing"

#val_med = LABELS['valence'].median()
#aro_med = LABELS['arousal'].median()

LABELS_SCH_VALID_PROC = LABELS_SCH_VALID.reset_index().assign(
    timestamp=lambda x: pd.to_datetime(x['scheduledTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ),
    valence_bin = lambda x: np.where(x['valence'] > 0, 1, 0),
    arousal_bin = lambda x: np.where(x['arousal'] > 0, 1, 0),
    emo_quad = lambda x: np.where(x['valence'] > 0,2,0) + np.where(x['arousal']>0,1,0),
    stress_bin = lambda x: np.where(x['stress'] > 0, 1, 0),
    disturbance_bin = lambda x: np.where(x['disturbance'] > 0, 1, 0)
).set_index(
    ['pcode', 'timestamp']
)
#valence_med_bin = lambda x: np.where(x['valence']>val_med,1,0),
#arousal_med_bin = lambda x: np.where(x['arousal']>aro_med,1,0),
#LABELS_SCH_VALID_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseValid.csv'))


LABELS_SCH_EXCL_PROC = LABELS_SCH_EXCL.reset_index().assign(
    timestamp=lambda x: pd.to_datetime(x['scheduledTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ),
    valence_bin = lambda x: np.where(x['valence'] > 0, 1, 0),
    arousal_bin = lambda x: np.where(x['arousal'] > 0, 1, 0),
    emo_quad = lambda x: np.where(x['valence'] > 0,2,0) + np.where(x['arousal']>0,1,0),
    stress_bin = lambda x: np.where(x['stress'] > 0, 1, 0),
    disturbance_bin = lambda x: np.where(x['disturbance'] > 0, 1, 0)
).set_index(
    ['pcode', 'timestamp']
)
#LABELS_SCH_EXCL_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponseExcluded.csv'))


LABELS_VOL_PROC = LABELS_VOL.reset_index().assign(
    responseTime=lambda x: pd.to_datetime(x['responseTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ),
    valence_bin = lambda x: np.where(x['valence'] > 0, 1, 0),
    arousal_bin = lambda x: np.where(x['arousal'] > 0, 1, 0),
    emo_quad = lambda x: np.where(x['valence'] > 0,2,0) + np.where(x['arousal']>0,1,0),
    stress_bin = lambda x: np.where(x['stress'] > 0, 1, 0),
    disturbance_bin = lambda x: np.where(x['disturbance'] > 0, 1, 0)
).set_index(
    ['pcode', 'responseTime']
)
#LABELS_VOL_PROC.to_csv(os.path.join(PATH_DATA, 'VoluntaryResponse.csv'))

LABELS_SCHED_PROC = LABELS_SCHED.reset_index().assign(
    timestamp=lambda x: pd.to_datetime(x['scheduledTime'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ),
    valence_bin = lambda x: np.where(x['valence'] > 0, 1, 0),
    arousal_bin = lambda x: np.where(x['arousal'] > 0, 1, 0),
    emo_quad = lambda x: np.where(x['valence'] > 0,2,0) + np.where(x['arousal']>0,1,0),
    stress_bin = lambda x: np.where(x['stress'] > 0, 1, 0),
    disturbance_bin = lambda x: np.where(x['disturbance'] > 0, 1, 0)
).set_index(
    ['pcode', 'timestamp']
)
#LABELS_SCHED_PROC.to_csv(os.path.join(PATH_DATA, 'ScheduledResponse.csv'))

#LABELS_PROC.head()

#%%

"""inst = LABELS_SCH_VALID_PROC.groupby('pcode').count().iloc[:, -1]
for c in [c for c in LABELS_SCH_VALID_PROC.columns if c.endswith('_bin') or c.endswith('_quad')]:
    print(f'- {c}:', summary(LABELS_SCH_VALID_PROC[c].astype(object)))
for c in [c for c in LABELS_SCH_EXCL_PROC.columns if c.endswith('_bin') or c.endswith('_quad')]:
    print(f'- {c}:', summary(LABELS_SCH_EXCL_PROC[c].astype(object)))
for c in [c for c in LABELS_VOL_PROC.columns if c.endswith('_bin') or c.endswith('_quad')]:
    print(f'- {c}:', summary(LABELS_VOL_PROC[c].astype(object)))
"""

# ## Sensor Data

# For each type of sensor data, we applied different preprocessing. Detailed decsription is provided in the paper.

# ### Execution

FUNC_SENSOR_PROC = {
    'Acceleration': proc._proc_acceleration,
    'AmbientLight': proc._proc_ambient_light,
    'Calorie': proc._proc_calories,
    'Distance': proc._proc_distance,
    'EDA': proc._proc_eda,
    'HR': proc._proc_hr,
    'RRI': proc._proc_rri,
    'SkinTemperature': proc._proc_skin_temperature,
    'StepCount': proc._proc_step_count,
    'UltraViolet': proc._proc_ultra_violet
}

FUNC_PHONE_PROC = {
    'ActivityEvent': proc._proc_activity_event,
    'ActivityTransition': proc._proc_activity_transition,
    'AppUsageEvent': proc._proc_app_usage,
    'BatteryEvent': proc._proc_battery,
    'CallEvent': proc._proc_call,
    'Connectivity': proc._proc_connectivity,
    'DataTraffic': proc._proc_data_traffic,
    'InstalledApp': proc._proc_installed_app,
    'Location': proc._proc_location,
    'MediaEvent': proc._proc_media_event,
    'MessageEvent': proc._proc_message_event,
    'WiFi': proc._proc_wifi,
    'ScreenEvent': proc._proc_screen,
    'RingerModeEvent': proc._proc_ringer_mode,
    'ChargeEvent': proc._proc_charge,
    'PowerSaveEvent': proc._proc_power_save,
    'OnOffEvent': proc._proc_on_off
}

"Execute pre-processing of sensor data"
if use_data == 'sensors':
    DATA_TYPES = SENSOR_TYPES
    FUNC_PROC = FUNC_SENSOR_PROC
elif use_data == 'phone':
    DATA_TYPES = PHONE_TYPES
    FUNC_PROC = FUNC_PHONE_PROC
else:
    DATA_TYPES = {**SENSOR_TYPES, **PHONE_TYPES}
    FUNC_PROC = {**FUNC_SENSOR_PROC, **FUNC_PHONE_PROC}

def _process(data_type: str,types_list: dict):
    log(f'Begin to processing data: {data_type}')
    
    abbrev = types_list[data_type]
    data_raw = _load_data(data_type,PATH_SENSOR)
    data_proc = FUNC_PROC[data_type](data_raw)
    result = dict()
    
    if type(data_proc) is dict:
        for k, v in data_proc.items():
            result[f'{abbrev}_{k}'] = v
    else:
        result[abbrev] = data_proc
        
    log(f'Complete processing data: {data_type}')
    return result

jobs = []  
for data_type in DATA_TYPES:
    job = _process(data_type,DATA_TYPES)
    jobs.append(job)

#%%
"Save the processed sensor data"
if use_data == 'sensors':
    edit_data = {'ACC_AXX': jobs[0]['ACC_AXX'],
                'ACC_AXY': jobs[0]['ACC_AXY'],
                'ACC_AXZ': jobs[0]['ACC_AXZ'],
                'ACC_MAG': jobs[0]['ACC_MAG'],
                'AML': jobs[1]['AML'],
                'CAL': jobs[2]['CAL'],
                'DST_DST': jobs[3]['DST_DST'],
                'DST_MOT': jobs[3]['DST_MOT'],
                'DST_PAC': jobs[3]['DST_PAC'],
                'DST_SPD': jobs[3]['DST_SPD'],
                'EDA': jobs[4]['EDA'],
                'HRT': jobs[5]['HRT'],
                'RRI': jobs[6]['RRI'],
                'SKT': jobs[7]['SKT'],
                'STP': jobs[8]['STP'],
                'ULV_INT': jobs[9]['ULV_INT'],
                'ULV_EXP': jobs[9]['ULV_EXP']
    }

elif use_data == 'phone':
    edit_data = {'ACE_UNK': jobs[0]['ACE_UNK'],
                 'ACE_FOT': jobs[0]['ACE_FOT'],
                 'ACE_WLK': jobs[0]['ACE_WLK'],
                 'ACE_VHC': jobs[0]['ACE_VHC'],
                 'ACE_RUN': jobs[0]['ACE_RUN'],
                 'ACE_BCC': jobs[0]['ACE_BCC'],
                 'ACE_TLT': jobs[0]['ACE_TLT'],
                 'ACT': jobs[1]['ACT'],
                 'APP_PAC': jobs[2]['APP_PAC'],
                 'APP_CAT': jobs[2]['APP_CAT'],
                 'BAT_LEV': jobs[3]['BAT_LEV'],
                 'BAT_STA': jobs[3]['BAT_STA'],
                 'BAT_TMP': jobs[3]['BAT_TMP'],
                 'CAE': jobs[4]['CAE'],
                 'CON': jobs[5]['CON'],
                 'DAT_RCV': jobs[6]['DAT_RCV'],
                 'DAT_SNT': jobs[6]['DAT_SNT'],
                 'INS_JAC': jobs[7]['INS_JAC'],
                 'LOC_CLS': jobs[8]['LOC_CLS'],
                 'LOC_DST': jobs[8]['LOC_DST'],
                 'MED_VID': jobs[9]['MED_VID'],
                 'MED_IMG': jobs[9]['MED_IMG'],
                 'MED_ALL': jobs[9]['MED_ALL'],
                 'MSG_SNT': jobs[10]['MSG_SNT'],
                 'MSG_RCV': jobs[10]['MSG_RCV'],
                 'MSG_ALL': jobs[10]['MSG_ALL'],
                 'WIF_COS': jobs[11]['WIF_COS'],
                 'WIF_EUC': jobs[11]['WIF_EUC'],
                 'WIF_MAN': jobs[11]['WIF_MAN'],
                 'WIF_JAC': jobs[11]['WIF_JAC'],
                 'SCR': jobs[12]['SCR'],
                 'RNG': jobs[13]['RNG'],
                 'CHG': jobs[14]['CHG'],
                 'PWS': jobs[15]['PWS'],
                 'ONF': jobs[16]['ONF']
    }

elif use_data == 'all':

    edit_data = {'ACC_AXX': jobs[0]['ACC_AXX'],
            'ACC_AXY': jobs[0]['ACC_AXY'],
            'ACC_AXZ': jobs[0]['ACC_AXZ'],
            'ACC_MAG': jobs[0]['ACC_MAG'],
            'AML': jobs[1]['AML'],
            'CAL': jobs[2]['CAL'],
            'DST_DST': jobs[3]['DST_DST'],
            'DST_MOT': jobs[3]['DST_MOT'],
            'DST_PAC': jobs[3]['DST_PAC'],
            'DST_SPD': jobs[3]['DST_SPD'],
            'EDA': jobs[4]['EDA'],
            'HRT': jobs[5]['HRT'],
            'RRI': jobs[6]['RRI'],
            'SKT': jobs[7]['SKT'],
            'STP': jobs[8]['STP'],
            'ULV_INT': jobs[9]['ULV_INT'],
            'ULV_EXP': jobs[9]['ULV_EXP'],
            'ACE_UNK': jobs[10]['ACE_UNK'],
                'ACE_FOT': jobs[10]['ACE_FOT'],
                'ACE_WLK': jobs[10]['ACE_WLK'],
                'ACE_VHC': jobs[10]['ACE_VHC'],
                'ACE_RUN': jobs[10]['ACE_RUN'],
                'ACE_BCC': jobs[10]['ACE_BCC'],
                'ACE_TLT': jobs[10]['ACE_TLT'],
                'ACT': jobs[11]['ACT'],
                'APP_PAC': jobs[12]['APP_PAC'],
                'APP_CAT': jobs[12]['APP_CAT'],
                'BAT_LEV': jobs[13]['BAT_LEV'],
                'BAT_STA': jobs[13]['BAT_STA'],
                'BAT_TMP': jobs[13]['BAT_TMP'],
                'CAE': jobs[14]['CAE'],
                'CON': jobs[15]['CON'],
                'DAT_RCV': jobs[16]['DAT_RCV'],
                'DAT_SNT': jobs[16]['DAT_SNT'],
                'INS_JAC': jobs[17]['INS_JAC'],
                'LOC_CLS': jobs[18]['LOC_CLS'],
                'LOC_DST': jobs[18]['LOC_DST'],
                'MED_VID': jobs[19]['MED_VID'],
                'MED_IMG': jobs[19]['MED_IMG'],
                'MED_ALL': jobs[19]['MED_ALL'],
                'MSG_SNT': jobs[20]['MSG_SNT'],
                'MSG_RCV': jobs[20]['MSG_RCV'],
                'MSG_ALL': jobs[20]['MSG_ALL'],
                'WIF_COS': jobs[21]['WIF_COS'],
                'WIF_EUC': jobs[21]['WIF_EUC'],
                'WIF_MAN': jobs[21]['WIF_MAN'],
                'WIF_JAC': jobs[21]['WIF_JAC'],
                'SCR': jobs[22]['SCR'],
                'RNG': jobs[23]['RNG'],
                'CHG': jobs[24]['CHG'],
                'PWS': jobs[25]['PWS'],
                'ONF': jobs[26]['ONF']
    }
    
dump(edit_data, os.path.join(PATH_INTERMEDIATE, use_data+'_proc.pkl'))


#dump(jobs, os.path.join(PATH_INTERMEDIATE, use_data+'_proc.pkl'))

del jobs, edit_data
gc.collect()

 
# In[84]:
"See sensors data overview"
#Uncomment below three lines to change data to dict
"""PATH_INTERMEDIATE = os.path.join(os.getcwd(),'intermediate')
from utils import load, dump
edit_data = {'ACC_AXX': DATA[0]['ACC_AXX'],
             'ACC_AXY': DATA[0]['ACC_AXY'],
             'ACC_AXZ': DATA[0]['ACC_AXZ'],
             'ACC_MAG': DATA[0]['ACC_MAG'],
             'AML': DATA[1]['AML'],
             'CAL': DATA[2]['CAL'],
             'DST_DST': DATA[3]['DST_DST'],
             'DST_MOT': DATA[3]['DST_MOT'],
             'DST_PAC': DATA[3]['DST_PAC'],
             'DST_SPD': DATA[3]['DST_SPD'],
             'EDA': DATA[4]['EDA'],
             'HRT': DATA[5]['HRT'],
             'RRI': DATA[6]['RRI'],
             'SKT': DATA[7]['SKT'],
             'STP': DATA[8]['STP'],
             'ULV_INT': DATA[9]['ULV_INT'],
             'ULV_EXP': DATA[9]['ULV_EXP']
}
dump(edit_data, os.path.join(PATH_INTERMEDIATE, 'sensors_proc.pkl'))
"""

DATA = load(os.path.join(PATH_INTERMEDIATE, use_data+'_proc.pkl'))
N_NUMERIC, N_CATEGORICAL = 0, 0

for k, v in DATA.items():
    if v.dtype.kind.isupper() or v.dtype.kind == 'b': 
        N_CATEGORICAL = N_CATEGORICAL + 1
    else:
        N_NUMERIC = N_NUMERIC + 1
        
    inst = v.groupby('pcode').count()
    sam = np.concatenate([
        (v.loc[(p,)].index.array - v.loc[(p,)].index.array.shift(1)).dropna().total_seconds()
        for p in v.index.get_level_values('pcode').unique()
    ])
    
    print('#'*5, k, '#'*5, )
    print('- # Inst.:', summary(inst))
    print('- Samp. period:', summary(sam))
    print('- Values', summary(v))
    print('')
    
    
print(f'# categorical data: {N_CATEGORICAL}/# numeric data: {N_NUMERIC}')
del DATA
gc.collect()
