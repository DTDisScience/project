# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def date_time(df: pd.DataFrame, column: str) -> pd.DataFrame:
    parsed = pd.to_datetime(df[column], format="%Y%m%d %H").dt
    sin_time = df[column].map(lambda x: np.sin(2 * np.pi * (int(x.split(" ")[1])) / 24))

    return pd.DataFrame(
        {
            "year": parsed.year,
            "month": parsed.month,
            "weekday": parsed.weekday,
            "day": parsed.day,
            "hour": parsed.hour,
            "sintime": sin_time,
        }
    )


def find_bdType(df1: pd.DataFrame, df2: pd.DataFrame):
    translation_dict = {
        '건물기타': 'other buildings',
        '공공': 'public',
        '대학교': 'university',
        '데이터센터': 'data center',
        '백화점및아울렛': 'department store and outlet',
        '병원': 'hospital',
        '상용': 'commercial',
        '아파트': 'apartment',
        '연구소': 'laboratory',
        '지식산업센터': 'knowledge industry center',
        '할인마트': 'discount mart',
        '호텔및리조트': 'hotel and resort'
    }
    bd_type = df2['건물유형'].map(lambda x: translation_dict[x])
    
    bd_type_list = []
    for i in range(1, 101):
        for j in range(0, int(len(df1)/100)):
            bd_type_list.append(bd_type[i-1])
    
    return bd_type_list


def typeHourDayConsumption(df: pd.DataFrame, group_columns: list, value_column: str, aggfunc):
    power_THDC_agg = pd.pivot_table(df, index=group_columns, values=value_column, aggfunc=aggfunc).reset_index()
    result_list = df.progress_apply(lambda x: round(power_THDC_agg.loc[(power_THDC_agg['building_type'] == x['building_type']) & 
                                                                       (power_THDC_agg['month'] == x['month']) &
                                                                       (power_THDC_agg['hour'] == x['hour']), value_column].values[0], 2), axis=1)
    return result_list

def find_variation(df_train: pd.DataFrame, df_test: pd.DataFrame, column: str):
    target_df = df_train  
    result_list = df_test.progress_apply(lambda x: target_df.loc[(target_df['building_number'] == x['building_number']) &
                                                                 (target_df['month'] == 8) &
                                                                 (target_df['weekday'] == x['weekday']) &
                                                                 (target_df['hour'] == x['hour']), column].unique()[0], axis=1)
    return result_list
    

def threeHourConsumption(df: pd.DataFrame, aggfun: str):
    if aggfun not in ['max', 'mean']:
        raise ValueError("Invalid value for 'condi'. Please use 'max' or 'mean'.")
    
    group_columns = ['building_number', 'weekday', 'month', 'hour']
    hour_filter = df['hour'].map(lambda x: [x-1, x, 0] if x == 23 else ([23, x, x+1] if x == 0 else [x-1, x, x+1]))
    df2 = df[:]
    df2['hour_filter'] = hour_filter
    
    if aggfun == 'max' :
        power_THC_agg = pd.pivot_table(df, index=group_columns, values='power', aggfunc=np.max).reset_index()
        result_list = df2.progress_apply(lambda x: round(power_THC_agg.loc[(power_THC_agg['building_number'] == x['building_number']) & 
                                                                           (power_THC_agg['weekday'] == x['weekday']) &
                                                                           (power_THC_agg['month'] == x['month']) &
                                                                           ((power_THC_agg['hour'] == x['hour_filter'][0])|
                                                                            (power_THC_agg['hour'] == x['hour_filter'][1])|
                                                                            (power_THC_agg['hour'] == x['hour_filter'][2])), 'power'].max(), 2), axis=1)
        return result_list
    else :
        power_THC_agg = pd.pivot_table(df, index=group_columns, values='power', aggfunc=np.mean).reset_index()
        result_list = df2.progress_apply(lambda x: round(power_THC_agg.loc[(power_THC_agg['building_number'] == x['building_number']) & 
                                                                           (power_THC_agg['weekday'] == x['weekday']) &
                                                                           (power_THC_agg['month'] == x['month']) &
                                                                           ((power_THC_agg['hour'] == x['hour_filter'][0])|
                                                                            (power_THC_agg['hour'] == x['hour_filter'][1])|
                                                                            (power_THC_agg['hour'] == x['hour_filter'][2])), 'power'].max(), 2), axis=1)
        return result_list




        
        
        


