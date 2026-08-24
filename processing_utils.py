import pandas as pd
import os
import pytz
import numpy as np
import scipy.spatial.distance as dist
from typing import Optional, Dict, Union
import pygeohash as geo
from datetime import timedelta
from collections import defaultdict

DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

def _load_data(
    name: str
,path_sensor) -> Optional[pd.DataFrame]:
    paths = [
        (d, os.path.join(path_sensor, d, f'{name}.csv'))
        for d in os.listdir(path_sensor)
        if d.startswith('P')
    ]
    return pd.concat(
        filter(
            lambda x: len(x.index), 
            [
                pd.read_csv(p).assign(pcode=pcode)
                for pcode, p in paths
                if os.path.exists(p)
            ]
        ), ignore_index=True
    ).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='ms', utc=True).dt.tz_convert(DEFAULT_TZ)
    ).set_index(
        ['pcode', 'timestamp']
    )


# AppUsageEvent.csv
def _proc_app_usage(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.loc[
        lambda x: x['type'].isin(['MOVE_TO_FOREGROUND', 'MOVE_TO_BACKGROUND']), :
    ].assign(
        packageName=lambda x: np.where(x['type'] == 'MOVE_TO_FOREGROUND', x['packageName'], None),
        category=lambda x: np.where(x['type'] == 'MOVE_TO_FOREGROUND', x['category'], None),
    )

    return {
        'PAC': data['packageName'].astype('object'),
        'CAT': data['category'].astype('object')
    }

# Connectivity.csv
def _proc_connectivity(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.assign(
        type=lambda x: np.where(x['isConnected'] == True, x['type'], 'DISCONNECTED')
    )

    return data['type'].astype('object')

# BatteryEvent.csv
def _proc_battery(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return {
        'LEV': data['level'].astype('float32'),
        'STA': data['status'].astype('object'),
        'TMP': data['temperature'].astype('float32')
    }
        
# CallEvent.csv
def _proc_call(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.loc[
        lambda x: x['duration'] > 0, :
    ]
    new_data = []

    for row in data.itertuples():
        pcode, timestamp = row.Index
        
        new_data.append({
            'pcode': pcode,
            'timestamp': timestamp,                
            'state': 'CALL',
        })
        new_data.append({
            'pcode': pcode,
            'timestamp': timestamp + timedelta(milliseconds=row.duration),
            'state': 'IDLE'
        })

    new_data = pd.DataFrame(new_data).set_index(
        ['pcode', 'timestamp']
    )

    return new_data['state'].astype('object')

# DataTraffic.csv
def _proc_data_traffic(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return {
        'RCV': data['rxKiloBytes'].astype('float32'),
        'SNT': data['txKiloBytes'].astype('float32')
    }

# RingerModeEvent.csv
def _proc_ringer_mode(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['type'].astype('object')

# ScreenEvent.csv
def _proc_screen(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['type'].astype('object')


# OnOffEvent.csv
def _proc_on_off(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['type'].astype('object')


# PowerSaveEvent.csv
def _proc_power_save(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['type'].astype('object')


# ChargeEvent.csv
def _proc_charge(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['type'].astype('object')


# Location.csv
def _proc_location(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    def _haversine(_lat1, _lat2, _lng1, _lng2) -> float:
        if np.isnan(_lat1) or np.isnan(_lat2) or np.isnan(_lng1) or np.isnan(_lng2):
            return 0.0

        _lat1_r, _lat2_r, _lng1_r, _lng2_r = np.radians(_lat1), np.radians(_lat2), np.radians(_lng1), np.radians(_lng2)
        _lat = _lat2_r - _lat1_r
        _lng = _lng2_r - _lng1_r
        _R = 6371008.8
        _d = np.sin(_lat * 0.5) ** 2 + np.cos(_lat1_r) * np.cos(_lat2_r) * np.sin(_lng * 0.5) ** 2
        return 2 * _R * np.arcsin(np.sqrt(_d))

    new_data = []
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            _latitude=lambda x: x['latitude'].shift(1),
            _longitude=lambda x: x['longitude'].shift(1),
            dist=lambda x: x.apply(
                lambda y: _haversine(y['latitude'], y['_latitude'], y['longitude'], y['_longitude']),
                axis=1
            ),
            cluster=lambda x: x.apply(
                lambda y: geo.encode(y['latitude'], y['longitude'], precision=7),
                axis=1
            ),
            pcode=pcode
        ).reset_index()
        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return {
        'CLS': new_data['cluster'].astype('object'),
        'DST': new_data['dist'].astype('float32')
    }

# ActivityEvent.csv
def _proc_activity_event(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return {
        'UNK': data['confidenceUnknown'].astype('float32'),
        'FOT': data['confidenceOnFoot'].astype('float32'),
        'WLK': data['confidenceWalking'].astype('float32'),
        'VHC': data['confidenceInVehicle'].astype('float32'),
        'BCC': data['confidenceOnBicycle'].astype('float32'),
        'RUN': data['confidenceRunning'].astype('float32'),
        'TLT': data['confidenceTilting'].astype('float32')
    }

# ActivityTransition.csv
def _proc_activity_transition(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.loc[
        lambda x: x['transitionType'].isin(['ENTER_WALKING', 'ENTER_STILL', 'ENTER_IN_VEHICLE', 'ENTER_ON_BICYCLE', 'ENTER_RUNNING']), :
    ].assign(
        type=lambda x: x['transitionType'].str.replace('ENTER_', '')
    )
    
    return data['type'].astype('object')

# WiFi.csv
def _proc_wifi(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            bssid=lambda x: x['bssid'].str.cat(x['frequency'].astype(str), sep='-')
        )
        t = sub.index.unique().array
        for cur_t, prev_t in zip(t, t.shift(1)):
            if cur_t is pd.NaT or prev_t is pd.NaT:
                continue

            prev = sub.loc[[prev_t], :]
            cur = sub.loc[[cur_t], :]
            intersect = np.intersect1d(prev['bssid'], cur['bssid'])
            union = np.union1d(prev['bssid'], cur['bssid'])
            w = np.repeat(1 / len(intersect), len(intersect)) if len(intersect) else 1.0
            prev_intersect = prev.loc[
                lambda x: x['bssid'].isin(intersect), :
            ].sort_values('bssid')
            cur_intersect = cur.loc[
                lambda x: x['bssid'].isin(intersect), :
            ].sort_values('bssid')
            prev_rssi = prev_intersect['rssi']
            cur_rssi = cur_intersect['rssi']

            new_data.append(dict(
                pcode=pcode,
                timestamp=cur_t,
                cosine=1 - dist.cosine(prev_rssi, cur_rssi) if len(intersect) > 0 else 0,
                euclidean=1 / (1 + dist.euclidean(prev_rssi, cur_rssi, w)) if len(intersect) > 0 else 0,
                manhattan=1 / (1 + dist.cityblock(prev_rssi, cur_rssi, w)) if len(intersect) > 0 else 0,
                jaccard = len(intersect) / len(union) if len(union) > 0 else 0
            ))
            
    new_data = pd.DataFrame(new_data).set_index(
        ['pcode', 'timestamp']
    )
    
    return {
        'COS': new_data['cosine'].astype('float32'),
        'EUC': new_data['euclidean'].astype('float32'),
        'MAN': new_data['manhattan'].astype('float32'),
        'JAC': new_data['jaccard'].astype('float32')
    }


# InstalledApp.csv
def _proc_installed_app(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(axis=0, level='timestamp')
        t = sub.index.unique().array
        for cur_t, prev_t in zip(t, t.shift(1)):
            if cur_t is pd.NaT or prev_t is pd.NaT:
                continue

            prev = sub.loc[[prev_t], :]
            cur = sub.loc[[cur_t], :]
            intersect = np.intersect1d(prev['packageName'], cur['packageName'])
            union = np.union1d(prev['packageName'], cur['packageName'])
            new_data.append(dict(
                pcode=pcode,
                timestamp=cur_t,
                jaccard = len(intersect) / len(union) if len(union) > 0 else 0
            ))
            
    new_data = pd.DataFrame(new_data).set_index(
        ['pcode', 'timestamp']
    )
    
    return {
       'JAC': new_data['jaccard'].astype('float32')
    }


# MediaEvent.csv
def _proc_media_event(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = defaultdict(list)
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        )

        video = sub.loc[
            lambda x: x['mimetype'].str.startswith('video'), :
        ].assign(
            event=1,
            pcode=pcode
        ).reset_index()

        image = sub.loc[
            lambda x: x['mimetype'].str.startswith('image'), :
        ].assign(
            event=1,
            pcode=pcode
        ).reset_index()

        media = sub.assign(
            event=1,
            pcode=pcode
        ).reset_index()

        new_data['VID'].append(video)
        new_data['IMG'].append(image)
        new_data['ALL'].append(media)

    return {
        k: pd.concat(
            v, axis=0, ignore_index=True
        ).set_index(
            ['pcode', 'timestamp']
        )['event'].astype('float32') 
        for k, v in new_data.items()
    }

# MessageEvent.csv
def _proc_message_event(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = defaultdict(list)
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        )

        sent = sub.loc[
            lambda x: x['messageBox'] == 'SENT', :
        ].assign(
            event=1,
            pcode=pcode
        ).reset_index()

        recv = sub.loc[
            lambda x: x['messageBox'] == 'INBOX', :
        ].assign(
            event=1,
            pcode=pcode
        ).reset_index()

        msg = sub.assign(
            event=1,
            pcode=pcode
        ).reset_index()

        new_data['SNT'].append(sent)
        new_data['RCV'].append(recv)
        new_data['ALL'].append(msg)

    return {
        k: pd.concat(
            v, axis=0, ignore_index=True
        ).set_index(
            ['pcode', 'timestamp']
        )['event'].astype('float32') 
        for k, v in new_data.items()
    }


# UltraViolet.csv
def _proc_ultra_violet(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []
    
    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            exposure=lambda x: (x['totalExposure'] - x['totalExposure'].shift(1)),
            pcode=pcode
        ).reset_index()

        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return {
        'INT': new_data['intensity'].astype('object'),
        'EXP': new_data['exposure'].dropna().astype('float32')
    }


# StepCount.csv
def _proc_step_count(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            steps=lambda x: (x['totalSteps'] - x['totalSteps'].shift(1)),
            pcode=pcode
        ).reset_index()
        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return new_data['steps'].dropna().astype('float32')
   

# Distance.csv
def _proc_distance(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            distance=lambda x: x['totalDistance'] - x['totalDistance'].shift(1),
            pcode=pcode
        ).reset_index()

        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return {
        'DST': new_data['distance'].dropna().astype('float32'),
        'MOT': new_data['motionType'].astype('object'),
        'PAC': new_data['pace'].astype('float32'),
        'SPD': new_data['speed'].astype('float32')
    }

# Calorie.csv
def _proc_calories(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    new_data = []

    for pcode in data.index.get_level_values('pcode').unique():
        sub = data.loc[(pcode, ), :].sort_index(
            axis=0, level='timestamp'
        ).assign(
            calories=lambda x: x['totalCalories'] - x['totalCalories'].shift(1),
            pcode=pcode
        ).reset_index()

        new_data.append(sub)

    new_data = pd.concat(new_data, axis=0, ignore_index=True).set_index(
        ['pcode', 'timestamp']
    )

    return new_data['calories'].dropna().astype('float32')           


# Acceleration.csv
def _proc_acceleration(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    data = data.assign(
        mag=lambda x: np.sqrt(np.square(x['x']) + np.square(x['y']) + np.square(x['z']))
    )

    #filter here

    return {
        'AXX': data['x'].astype('float32'),
        'AXY': data['y'].astype('float32'),
        'AXZ': data['z'].astype('float32'),
        'MAG': data['mag'].astype('float32')
    }

# SkinTemperature.csv
def _proc_skin_temperature(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    
    #maybe filter here

    return data['temperature'].astype('float32')

# RRI.csv
def _proc_rri(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    
    #maybe filter here

    return data['interval'].astype('float32')


# AmbientLight.csv
def _proc_ambient_light(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['brightness'].astype('float32')
    

# HR.csv
def _proc_hr(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['bpm'].astype('float32')
    

# EDA.csv
def _proc_eda(data: pd.DataFrame) -> Union[pd.Series, Dict[str, pd.Series]]:
    return data['resistance'].astype('float32')
