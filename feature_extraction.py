import numpy as np
import pandas as pd
from typing import Dict, Callable, Union, Tuple, List, Optional, Iterable
from datetime import timedelta as td
from datetime import datetime
from scipy import stats
#import ray
import warnings
import time
from utils import log
import matplotlib.pyplot as plt
from scipy import signal    

DATE_FORMAT = "%Y-%m-%d %H:%M:%S%z" 

def _safe_na_check(_v):
    _is_nan_inf = False
    
    try:
        _is_nan_inf = np.isnan(_v) or np.isinf(_v)
    except:
        _is_nan_inf = False
    
    return _is_nan_inf or _v is None

def _butter_lowpass(sig,cutoff, fs, ptype, order=5):
    sos = signal.butter(order, cutoff*fs, ptype, fs=fs, output='sos')
    return signal.sosfilt(sos, sig)

def _extract_all_labels(pid: str,
        data: Dict[str, pd.Series],
        label: Union[pd.Series,pd.DataFrame],
        label_values: Union[List[str],Dict[str, List[int]]],
        window_data: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],
        window_label: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],        
        categories: Dict[str, Optional[List[any]]] = None,        
        constant_features: Dict[str, any] = None,
        resample_s: Dict[str, float] = None,
        label_based_features: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    _s = time.time()
    log(f"Begin feature extraction on {pid}'s data.")

    categories = categories or dict()
    constant_features = constant_features or dict()
    resample_s = resample_s or dict()

    X, y, date_times = [], [], []
    "Extract features from sensor data for every given label"
    for timestamp in label.index:
        row = dict()

        label_cur = label.loc[timestamp]
        date = datetime.strptime(timestamp, DATE_FORMAT)
        #t = timestamp - td(milliseconds=1)
        t = date - td(milliseconds=1) #convert to datetime from str
        #t = date.strftime(DATE_FORMAT) #convert to str from datetime
        #t = t[:-2] + ':' + t[-2:] #add colon to timezone offset

        # Features relevant to participants' info
        for d_key, d_val in constant_features.items():
            row[d_key] = d_val
        
        # Features from sensor data
        for d_key, d_val in data.items():
            is_numeric = d_key not in categories
            cats = categories.get(d_key) or list()
            d_val = d_val.sort_index()

            # Features relevant to latest value of a given data
            # These features are extracted only for bounded categorical data and numerical data.
            if is_numeric or cats:
                try:
                    v = d_val.loc[:t].iloc[-1]
                except (KeyError, IndexError):
                    v = 0

                if is_numeric:
                    row[f'{d_key}#VAL'] = v
                else:
                    for c in cats:
                        row[f'{d_key}#VAL={c}'] = v == c
            # Features relevant to duration since the latest state change.
            # These features are only for categorical data.
            # In addition, duration since a given state is set recently is considered, 
            # that are available only at bounded categorical data.
            if not is_numeric:
                try:
                    v = d_val.loc[:t]
                    row[f'{d_key}#DSC'] = (t - v.index[-1]).total_seconds() if len(v) else -1.0
                    for c in cats:
                        v_sub = v.loc[lambda x: x == c].index
                        row[f'{d_key}#DSC={c}'] = (t - v_sub[-1]).total_seconds() if len(v_sub) else -1.0
                except (KeyError, IndexError):
                    row[f'{d_key}#DSC'] = -1.0
                    for c in cats:
                        row[f'{d_key}#DSC={c}'] = -1.0

            # Features extracted from time-windows
            # These features requires resampling and imputation on each data.
            sample_rate = resample_s.get(d_key) or 1
            d_val_res = d_val.resample(f'{sample_rate}S', origin='start')
            if is_numeric:
                d_val_res = d_val_res.mean().interpolate(method='linear').dropna()
            else:
                d_val_res = d_val_res.ffill().dropna()

            #Also apply a filtering for sensor signals - Excluded for now
            """if is_numeric:
                LPB = filt_freqs.get(d_key) or None
                fs = 1/sample_rate
                if LPB is not None:
                    d_val_filt = pd.Series(_butter_lowpass(d_val_res,LPB, fs, 'lp', order=5))
                    d_val_filt.index = d_val_res.index"""

            for w_key, w_val in window_data.items():
                w_val = w_val(t) if isinstance(w_val, Callable) else w_val
                try:
                    v = d_val_res.loc[t - td(seconds=w_val):t]
                except (KeyError, IndexError):
                    continue
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')

                    if is_numeric:
                        hist, _ = np.histogram(v, bins='doane', density=False)
                        std = np.sqrt(np.var(v, ddof=1)) if len(v) > 1 else 0
                        v_norm = (v - np.mean(v)) / std if std != 0 else np.zeros(len(v))
                        #Save the normalized signal v_norm
                        row[f'{d_key}#AVG#{w_key}'] = np.mean(v) # Sample mean
                        row[f'{d_key}#STD#{w_key}'] = std # Sample standard deviation
                        row[f'{d_key}#SKW#{w_key}'] = stats.skew(v, bias=False) # Sample skewness
                        row[f'{d_key}#KUR#{w_key}'] = stats.kurtosis(v, bias=False) # Sample kurtosis
                        row[f'{d_key}#ASC#{w_key}'] = np.sum(np.abs(np.diff(v))) # Abstract sum of changes
                        row[f'{d_key}#BEP#{w_key}'] = stats.entropy(hist) # Binned entropy
                        row[f'{d_key}#MED#{w_key}'] = np.median(v) # Median
                        row[f'{d_key}#TSC#{w_key}'] = np.sqrt(np.sum(np.power(np.diff(v_norm), 2))) # Timeseries complexity
                    else:
                        cnt = v.value_counts()
                        val, sup = cnt.index, cnt.values
                        hist = {k: v for k, v in zip(val, sup)}
                        
                        # Information Entropy
                        row[f'{d_key}#ETP#{w_key}'] = stats.entropy(sup / len(v))
                        
                        # Abs. Sum of Changes
                        row[f'{d_key}#ASC#{w_key}'] = np.sum(v.values[1:] != v.values[:-1]) 

                        if len(cats) == 2: # Dichotomous categorical data
                            c = cats[0]
                            row[f'{d_key}#DUR#{w_key}'] = hist[c] / len(v) if c in hist else 0
                        else:
                            for c in cats:
                                row[f'{d_key}#DUR={c}#{w_key}'] = hist[c] / len(v) if c in hist else 0
        
        # Features relevant to time
        day_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][t.isoweekday() - 1]
        is_weekend = 'Y' if t.isoweekday() > 5 else 'N'
        hour = t.hour

        if 6 <= hour < 9:
            hour_name = 'DAWN'
        elif 9 <= hour < 12:
            hour_name = 'MORNING'
        elif 12 <= hour < 15:
            hour_name = 'AFTERNOON'
        elif 15 <= hour < 18:
            hour_name = 'LATE_AFTERNOON'
        elif 18 <= hour < 21:
            hour_name = 'EVENING'
        elif 21 <= hour < 24:
            hour_name = 'NIGHT'
        else:
            hour_name = 'MIDNIGHT'
        
        for d in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']:
            row[f'ESM#DOW={d}'] = d == day_of_week

        for d in ['Y', 'N']:
            row[f'ESM#WKD={d}'] = d == is_weekend
        
        for d in ['DAWN', 'MORNING', 'AFTERNOON', 'LATE_AFTERNOON', 'EVENING', 'NIGHT', 'MIDNIGHT']:
            row[f'ESM#HRN={d}'] = d == hour_name
        
        if label_based_features:
            # Features extracted from previous respones behavior
            for w_key, w_val in window_label.items():
                w_val = w_val(t) if isinstance(w_val, Callable) else w_val

                for key in label_values.keys():
                    lab_vals = label_values.get(key)
                    "Index label with str format of timestamp"
                    tstart = t - td(seconds=w_val)
                    tstartstr = tstart.strftime(DATE_FORMAT) #convert to str from datetime
                    tstartstr = tstartstr[:-2] + ':' + tstartstr[-2:]
                    tstr = t.strftime(DATE_FORMAT) 
                    try:
                        #Throw an error if the starting time is before the first label
                        if datetime.strptime(label.index[0], DATE_FORMAT) > tstart: 
                            raise IndexError('Starting time is before the first label')
                        v = label.loc[tstartstr:tstr,key] #will throw error if N/A (first labels)

                        if len(lab_vals) <= 2: # Binary classification
                            row[f'ESM#LIK#{w_key}#{key}'] = np.sum(v == lab_vals[0]) / len(v) if len(v) > 0 else 0
                        else:
                            for l in lab_vals:
                                row[f'ESM#LIK={l}#{w_key}#{key}'] = np.sum(v == l) / len(v) if len(v) > 0 else 0
                    
                    except (KeyError, IndexError, TypeError):
                        if len(lab_vals) <= 2: 
                            row[f'ESM#LIK#{w_key}#{key}'] = 0
                        else:
                            for l in lab_vals:
                                row[f'ESM#LIK={l}#{w_key}#{key}'] = 0
        
        #Replace NaNs with 0s in the entire feature vector                     
        row = {
            k: 0.0 if _safe_na_check(v) else v
            for k, v in row.items()
        }
        X.append(row)
        y.append(label_cur)
        date_times.append(timestamp)
    
    log(f"Complete feature extraction on {pid}'s data ({time.time() - _s:.2f} s).")
    X, y, group, date_times = pd.DataFrame(X), np.asarray(y), np.repeat(pid, len(y)), np.asarray(date_times)
    return X, y, group, date_times


def _extract(
        pid: str,
        data: Dict[str, pd.Series],
        label: Union[pd.Series,pd.DataFrame],
        label_values: List[str],
        window_data: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],
        window_label: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],        
        categories: Dict[str, Optional[List[any]]] = None,        
        constant_features: Dict[str, any] = None,
        resample_s: Dict[str, float] = None,
        label_based_features: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _s = time.time()
    log(f"Begin feature extraction on {pid}'s data.")

    categories = categories or dict()
    constant_features = constant_features or dict()
    resample_s = resample_s or dict()

    X, y, date_times = [], [], []
    "Extract features from sensor data for every given label"
    for timestamp in label.index:
        row = dict()

        label_cur = label.at[timestamp]
        date = datetime.strptime(timestamp, DATE_FORMAT)
        #t = timestamp - td(milliseconds=1)
        t = date - td(milliseconds=1) #convert to datetime from str
        #t = tdate.strftime(date_format) #convert to str from datetime
        #t = t[:-2] + ':' + t[-2:] #add colon to timezone offset

        # Features relevant to participants' info
        for d_key, d_val in constant_features.items():
            row[d_key] = d_val
        
        # Features from sensor data
        for d_key, d_val in data.items():
            is_numeric = d_key not in categories
            cats = categories.get(d_key) or list()
            d_val = d_val.sort_index()

            # Features relevant to latest value of a given data
            # These features are extracted only for bounded categorical data and numerical data.
            if is_numeric or cats:
                try:
                    v = d_val.loc[:t].iloc[-1]
                except (KeyError, IndexError):
                    v = 0

                if is_numeric:
                    row[f'{d_key}#VAL'] = v
                else:
                    for c in cats:
                        row[f'{d_key}#VAL={c}'] = v == c
            # Features relevant to duration since the latest state change.
            # These features are only for categorical data.
            # In addition, duration since a given state is set recently is considered, 
            # that are available only at bounded categorical data.
            if not is_numeric:
                try:
                    v = d_val.loc[:t]
                    row[f'{d_key}#DSC'] = (t - v.index[-1]).total_seconds() if len(v) else -1.0
                    for c in cats:
                        v_sub = v.loc[lambda x: x == c].index
                        row[f'{d_key}#DSC={c}'] = (t - v_sub[-1]).total_seconds() if len(v_sub) else -1.0
                except (KeyError, IndexError):
                    row[f'{d_key}#DSC'] = -1.0
                    for c in cats:
                        row[f'{d_key}#DSC={c}'] = -1.0

            # Features extracted from time-windows
            # These features requires resampling and imputation on each data.
            sample_rate = resample_s.get(d_key) or 1
            d_val_res = d_val.resample(f'{sample_rate}S', origin='start')
            if is_numeric:
                d_val_res = d_val_res.mean().interpolate(method='linear').dropna()
            else:
                d_val_res = d_val_res.ffill().dropna()

            #Also apply a filtering for sensor signals - Excluded for now
            """if is_numeric:
                LPB = filt_freqs.get(d_key) or None
                fs = 1/sample_rate
                if LPB is not None:
                    d_val_filt = pd.Series(_butter_lowpass(d_val_res,LPB, fs, 'lp', order=5))
                    d_val_filt.index = d_val_res.index"""

            for w_key, w_val in window_data.items():
                w_val = w_val(t) if isinstance(w_val, Callable) else w_val
                try:
                    v = d_val_res.loc[t - td(seconds=w_val):t]
                except (KeyError, IndexError):
                    continue
                
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')

                    if is_numeric:
                        hist, _ = np.histogram(v, bins='doane', density=False)
                        std = np.sqrt(np.var(v, ddof=1)) if len(v) > 1 else 0
                        v_norm = (v - np.mean(v)) / std if std != 0 else np.zeros(len(v))
                        #Save the normalized signal v_norm
                        row[f'{d_key}#AVG#{w_key}'] = np.mean(v) # Sample mean
                        row[f'{d_key}#STD#{w_key}'] = std # Sample standard deviation
                        row[f'{d_key}#SKW#{w_key}'] = stats.skew(v, bias=False) # Sample skewness
                        row[f'{d_key}#KUR#{w_key}'] = stats.kurtosis(v, bias=False) # Sample kurtosis
                        row[f'{d_key}#ASC#{w_key}'] = np.sum(np.abs(np.diff(v))) # Abstract sum of changes
                        row[f'{d_key}#BEP#{w_key}'] = stats.entropy(hist) # Binned entropy
                        row[f'{d_key}#MED#{w_key}'] = np.median(v) # Median
                        row[f'{d_key}#TSC#{w_key}'] = np.sqrt(np.sum(np.power(np.diff(v_norm), 2))) # Timeseries complexity
                    else:
                        cnt = v.value_counts()
                        val, sup = cnt.index, cnt.values
                        hist = {k: v for k, v in zip(val, sup)}
                        
                        # Information Entropy
                        row[f'{d_key}#ETP#{w_key}'] = stats.entropy(sup / len(v))
                        
                        # Abs. Sum of Changes
                        row[f'{d_key}#ASC#{w_key}'] = np.sum(v.values[1:] != v.values[:-1]) 

                        if len(cats) == 2: # Dichotomous categorical data
                            c = cats[0]
                            row[f'{d_key}#DUR#{w_key}'] = hist[c] / len(v) if c in hist else 0
                        else:
                            for c in cats:
                                row[f'{d_key}#DUR={c}#{w_key}'] = hist[c] / len(v) if c in hist else 0
        
        # Features relevant to time
        day_of_week = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][t.isoweekday() - 1]
        is_weekend = 'Y' if t.isoweekday() > 5 else 'N'
        hour = t.hour

        if 6 <= hour < 9:
            hour_name = 'DAWN'
        elif 9 <= hour < 12:
            hour_name = 'MORNING'
        elif 12 <= hour < 15:
            hour_name = 'AFTERNOON'
        elif 15 <= hour < 18:
            hour_name = 'LATE_AFTERNOON'
        elif 18 <= hour < 21:
            hour_name = 'EVENING'
        elif 21 <= hour < 24:
            hour_name = 'NIGHT'
        else:
            hour_name = 'MIDNIGHT'
        
        for d in ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']:
            row[f'ESM#DOW={d}'] = d == day_of_week

        for d in ['Y', 'N']:
            row[f'ESM#WKD={d}'] = d == is_weekend
        
        for d in ['DAWN', 'MORNING', 'AFTERNOON', 'LATE_AFTERNOON', 'EVENING', 'NIGHT', 'MIDNIGHT']:
            row[f'ESM#HRN={d}'] = d == hour_name
        
        if label_based_features:
            # Features extracted from previous responses behavior
            for w_key, w_val in window_label.items():
                w_val = w_val(t) if isinstance(w_val, Callable) else w_val

                "Index label with str format of timestamp"
                tstart = t - td(seconds=w_val)
                tstartstr = tstart.strftime(DATE_FORMAT) #convert to str from datetime
                tstartstr = tstartstr[:-2] + ':' + tstartstr[-2:]
                tstr = t.strftime(DATE_FORMAT) 
                try:
                    #Throw an error if the starting time is before the first label
                    if datetime.strptime(label.index[0], DATE_FORMAT) > tstart: 
                        raise IndexError('Starting time is before the first label')
                    v = label.loc[tstartstr:tstr] #will throw error if N/A (first labels)

                    if len(label_values) <= 2: # Binary classification
                        row[f'ESM#LIK#{w_key}'] = np.sum(v == label_values[0]) / len(v) if len(v) > 0 else 0
                    else:
                        for l in label_values:
                            row[f'ESM#LIK={l}#{w_key}'] = np.sum(v == l) / len(v) if len(v) > 0 else 0
                except (KeyError, IndexError, TypeError):
                    if len(label_values) <= 2: 
                        row[f'ESM#LIK#{w_key}'] = 0
                    else:
                        for l in label_values:
                            row[f'ESM#LIK={l}#{w_key}'] = 0

        #Replace NaNs with 0s in the entire feature vector       
        row = {
            k: 0.0 if _safe_na_check(v) else v
            for k, v in row.items()
        }
        X.append(row)
        y.append(label_cur)
        date_times.append(timestamp)
    
    log(f"Complete feature extraction on {pid}'s data ({time.time() - _s:.2f} s).")
    X, y, group, date_times = pd.DataFrame(X), np.asarray(y), np.repeat(pid, len(y)), np.asarray(date_times)
    return X, y, group, date_times


def extract(
        pids: Iterable[str],
        data: Dict[str, pd.Series],
        label: Union[pd.Series,pd.DataFrame],
        label_values: Union[List[str],Dict[str, List[int]]],
        window_data: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],
        window_label: Dict[str, Union[int, Callable[[pd.Timestamp], int]]],        
        categories: Dict[str, Optional[List[any]]] = None,        
        constat_features: Dict[str, Dict[str, any]] = None,
        resample_s: Dict[str, float] = None,
        label_based_features: bool = True,
        with_ray: bool=False
):
    #if with_ray and not ray.is_initialized():
    #    raise EnvironmentError('Ray should be initialized if "with_ray" is set as True.')
    
    #func = ray.remote(_extract).remote if with_ray else _extract
    func = _extract if isinstance(label,pd.Series) else _extract_all_labels
    if not isinstance(label,pd.DataFrame) and func == _extract_all_labels:
        raise TypeError('label should be a pandas DataFrame')
    jobs = []

    for pid in pids:
        d = dict()

        for k, v in data.items():
            try:
                d[k] = v.loc[(pid, )]
            except (KeyError, IndexError):
                pass

        job = func(
            pid=pid, data=d, label=label.loc[(pid,)], 
            label_values=label_values, 
            window_data=window_data, 
            window_label=window_label,
            categories=categories, 
            constant_features=constat_features[pid],
            resample_s=resample_s,
            label_based_features=label_based_features
        )
        jobs.append(job)
    
    #jobs = ray.get(jobs) if with_ray else jobs

    X = pd.concat([x for x, _, _, _ in jobs], axis=0, ignore_index=True)
    y = np.concatenate([x for _, x, _, _ in jobs], axis=0)
    group = np.concatenate([x for _, _, x, _ in jobs], axis=0)
    date_times_str = list(np.concatenate([x for _, _, _, x in jobs], axis=0))
    date_times = pd.DatetimeIndex([datetime.strptime(x,DATE_FORMAT) for x in date_times_str])

    t_s = date_times.min().normalize().timestamp()
    t_norm = np.asarray(list(map(lambda x: x.timestamp() - t_s, date_times)))
    
    C, DTYPE = X.columns, X.dtypes
    
    X = X.fillna({
        **{c: False for c in C[(DTYPE == object) | (DTYPE == bool)]},
        **{c: 0.0 for c in C[(DTYPE != object) & (DTYPE != bool)]},
    }).astype({
        **{c: 'bool' for c in C[(DTYPE == object) | (DTYPE == bool)]},
        **{c: 'float32' for c in C[(DTYPE != object) & (DTYPE != bool)]},
    })
        
    return X, y, group, t_norm, date_times_str
