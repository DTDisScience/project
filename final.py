# -*- coding: utf-8 -*-
import utils
import graph
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# =============================================================================
# 원본 데이터 불러오기
# =============================================================================
df_train_원본 = pd.read_csv('train.csv')
df_building_info_원본 = pd.read_csv('building_info.csv')
df_test_원본 = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

# step 1) num_date_time, 일조(hr), 일사(MJ/m2) 컬럼 드랍
df_train_원본 = df_train_원본.drop(['num_date_time', '일조(hr)', '일사(MJ/m2)'], axis=1)
df_test_원본 = df_test_원본.drop(['num_date_time'], axis=1)

# step 2) 전력소비량(kWh) 컬럼 별도 분리
train_y = df_train_원본['전력소비량(kWh)']
df_train_원본 = df_train_원본.drop(['전력소비량(kWh)'], axis=1)

# step 3) date_time메서드 및 sintime 적용을 위한 train/test concat
df_total = pd.concat([df_train_원본, df_test_원본])
result_df = utils.date_time(df_total, '일시')
# result_df 결과
#        year  month  weekday  day  hour   sintime
# 0      2022      6        2    1     0  0.000000
# 1      2022      6        2    1     1  0.258819
# 2      2022      6        2    1     2  0.500000
# 3      2022      6        2    1     3  0.707107
# 4      2022      6        2    1     4  0.866025
#     ...    ...      ...  ...   ...       ...
# 16795  2022      8        2   31    19 -0.965926
# 16796  2022      8        2   31    20 -0.866025
# 16797  2022      8        2   31    21 -0.707107
# 16798  2022      8        2   31    22 -0.500000
# 16799  2022      8        2   31    23 -0.258819
# =============================================================================
df_train = pd.concat([df_train_원본, result_df[:len(df_train_원본)], train_y], axis=1)
df_test = pd.concat([df_test_원본, result_df[len(df_train_원본):]], axis=1)

# step 4) 일시 컬럼 드랍 후 컬럼 정렬하기 & 컬럼명 영어로 변경
df_train = df_train.drop('일시', axis=1)
df_test = df_test.drop('일시', axis=1)

df_train1 = df_train.iloc[:, [0,5,6,7,8,9,10,1,2,3,4,11]]
df_test1 = df_test.iloc[:, [0,5,6,7,8,9,10,1,2,3,4]]

df_train1.columns = ['building_number', 'year', 'month', 'weekday', 'day',
                     'hour', 'sintime', 'temperature', 'rain', 'wind', 'humi', 'power']
df_test1.columns = ['building_number', 'year', 'month', 'weekday', 'day',
                    'hour', 'sintime','temperature', 'rain', 'wind', 'humi']


# step 5) 결측치 처리
df_train1.info()     # 강수량(160069개==>0), 풍속(19개), 습도(9개) 결측치 있음
df_train1['rain'] = df_train1['rain'].fillna(0)
df_train1['wind'].fillna(round(df_train1['wind'].mean(), 2), inplace=True)
df_train1['humi'].fillna(round(df_train1['humi'].mean(), 2), inplace=True)
df_test1.info()      # 결측치 없음

# =============================================================================
# building_info 이상치 처리
# =============================================================================
df_bd_info = df_building_info_원본[:]

# step 1) 아파트 냉방면적 결측치 처리(냉방면적 0)
imsi_apt_df = df_bd_info.loc[df_bd_info['건물유형'] == '아파트', ['연면적(m2)', '냉방면적(m2)']]
imsi_apt_df['ratio'] = list(map(lambda x, y : y / x, imsi_apt_df['연면적(m2)'], imsi_apt_df['냉방면적(m2)']))
mean_ratio_apt = imsi_apt_df.iloc[[0,1,2,3,6], 2].mean()
df_bd_info.loc[df_bd_info['건물번호'] == 65, '냉방면적(m2)'] = df_bd_info.loc[df_bd_info['건물번호'] == 65, '연면적(m2)'].values * mean_ratio_apt
df_bd_info.loc[df_bd_info['건물번호'] == 66, '냉방면적(m2)'] = df_bd_info.loc[df_bd_info['건물번호'] == 66, '연면적(m2)'].values * mean_ratio_apt
df_bd_info.loc[df_bd_info['건물번호'] == 68, '냉방면적(m2)'] = df_bd_info.loc[df_bd_info['건물번호'] == 68, '연면적(m2)'].values * mean_ratio_apt

# step 2) 지식산업센터 냉방면적 결측치 처리 (냉방면적 1, 239)
imsi_know_df = df_bd_info.loc[df_bd_info['건물유형'] == '지식산업센터', ['연면적(m2)', '냉방면적(m2)']]
imsi_know_df['ratio'] = list(map(lambda x, y : y / x, imsi_know_df['연면적(m2)'], imsi_know_df['냉방면적(m2)']))
mean_ratio_know = imsi_know_df.iloc[[1,2,4,5,6,7], 2].mean()
df_bd_info.loc[df_bd_info['건물번호'] == 77, '냉방면적(m2)'] = df_bd_info.loc[df_bd_info['건물번호'] == 77, '연면적(m2)'].values * mean_ratio_know
df_bd_info.loc[df_bd_info['건물번호'] == 80, '냉방면적(m2)'] = df_bd_info.loc[df_bd_info['건물번호'] == 80, '연면적(m2)'].values * mean_ratio_know


# =============================================================================
# 건물유형 컬럼 추가
# =============================================================================
df_train1['building_type'] = utils.find_bdType(df_train1, df_bd_info)
df_test1['building_type'] = utils.find_bdType(df_test1, df_bd_info)


# =============================================================================
# 건물유형별 일별 전력소비량 패턴 그래프로 보기
# =============================================================================
graph.dayCounsumption(df_train1, 220000)

# 데이터센터와 대학교를 제외하고 다시 그래프를 그려보면
rest_df_train1 = df_train1.loc[(df_train1['building_type'] != 'data center') & (df_train1['building_type'] != 'university'), :]
graph.dayCounsumption(rest_df_train1, 100000)

# 해당 그래프를 통해 건물유형에 따라 
# 평일 주말 전력소비 패턴과 그 소비량에 차이가 있음을 확인

# 건물유형, 건물번호별 일 평균 소비 패턴그래프
# ==> 시간대별 사용량에 유사한 패턴을 보임을 확인
for i in df_train1['building_type'].unique() :
    graph.typeHourConsumption(df_train1, i)


# =============================================================================
# 독립변수 추가하기
# 1. 건물유형별, 월별, 시간대별 전력사용량 평균, 표준편차, 최대사용량
#    (단, 모든 건물이 6월 대비 7,8월 전력사용량이 급증하므로
#     월별로 산출할 것)

# 2. 각 건물별, 월별, 요일별, 앞뒤 세시간 평균, 최대 전력사용량
# =============================================================================
# =============================================================================
# Train data 처리
# =============================================================================
# step 1) 건물유형별, 월별, 시간대별 전력사용량 평균, 표준편차
# ex) university의 6,7,8월 각 시간대별 전력사용량 평균, 표준편차, 최대사용량
mean_l = utils.typeHourDayConsumption(df_train1, ['building_type', 'month', 'hour'], 'power', np.mean)
std_l = utils.typeHourDayConsumption(df_train1, ['building_type', 'month', 'hour'], 'power', np.std)
max_l = utils.typeHourDayConsumption(df_train1, ['building_type', 'month', 'hour'], 'power', np.max)

# step 2) 각 건물별, 월별, 요일별, 앞뒤 세시간 평균, 최대 전력사용량
# ex) university의 6,7,8월 0시 =>  university의 6,7,8월 23시, 0시, 1시 전력사용량의 평균 & 최대
three_hour_mean = utils.threeHourConsumption(df_train1, 'mean')
three_hour_max = utils.threeHourConsumption(df_train1, 'max')

df_train1['typemonthhour_mean'] = mean_l
df_train1['typemonthhour_std'] = std_l
df_train1['typemonthhour_max'] = max_l
df_train1['threehour_mean'] = three_hour_mean
df_train1['threehour_max'] = three_hour_max
df_train1


# =============================================================================
# Test data 처리
# test 데이터는 모두 8월 데이터!
# train의 8월 데이터에서 해당 조건에 맞는 데이터를 가져와서 추가
# =============================================================================
# step 1) 건물유형별, 월별, 시간대별 전력사용량 평균, 표준편차 최대사용량 추가
tmh_mean = utils.find_variation(df_train1, df_test1, 'typemonthhour_mean')
tmh_std = utils.find_variation(df_train1, df_test1, 'typemonthhour_std')
tmh_max = utils.find_variation(df_train1, df_test1, 'typemonthhour_max')

# step 2) 각 건물별, 월별, 요일별, 앞뒤 세시간 평균, 최대 전력사용량
th_mean = utils.find_variation(df_train1, df_test1, 'threehour_mean')
th_max = utils.find_variation(df_train1, df_test1, 'threehour_max')

df_test1['typemonthhour_mean'] = tmh_mean
df_test1['typemonthhour_std'] = tmh_std
df_test1['typemonthhour_max'] = tmh_max
df_test1['threehour_mean'] = th_mean
df_test1['threehour_max'] = th_max
df_test1


# =============================================================================
# 독립변수 추가하기
# 4. 불쾌지수(TDI)
# 5. 온도, 습도 변화량
# =============================================================================
# step 4) 불쾌지수(TDI)
def DI_function(x, y):
    discomfort_index = round((1.8 * x) - (0.55 * ((1 - y)/100) * (1.8 * x - 26)) + 32, 2)
    return discomfort_index

train_discomfort_index_list = list(map(DI_function, df_train1['temperature'], df_train1['humi']))
test_discomfort_index_list = list(map(DI_function, df_test1['temperature'], df_test1['humi']))

df_train1['tdi'] = train_discomfort_index_list
df_test1['tdi'] = test_discomfort_index_list


# step 5) 온도 변화량
df_train1['temperature_change'] = df_train1['temperature'].diff()
df_train1['temperature_change'].fillna(0, inplace=True)

df_test1['temperature_change'] = df_test1['temperature'].diff()
df_test1['temperature_change'].fillna(df_train1['temperature'][len(df_train1)-1] - df_test1['temperature'][0], inplace=True)

# step 6) 습도 변화량
df_train1['humi_change'] = df_train1['humi'].diff()
df_train1['humi_change'].fillna(0, inplace=True)

df_test1['humi_change'] = df_test1['humi'].diff()
df_test1['humi_change'].fillna(df_train1['humi'][len(df_train1)-1] - df_test1['humi'][0], inplace=True)


# =============================================================================
# SMAPE 평가지표 설정
# =============================================================================
def smape(
    y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
):
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(
        np.abs(y_true) + np.abs(y_pred), epsilon
    )
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None
    return -np.average(output_errors, weights=multioutput)
SMAPE = make_scorer(smape, greater_is_better=False)


# =============================================================================
# 건물유형별 최적 독립변수 조합에 따른 최적 파라미터 값 출력
# =============================================================================
m_xgb = XGBRegressor()
param_grid = {
    'n_estimators' : [100, 500, 1000],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth' : [3, 5, 7],
    'colsample_bytree' : [0.7, 0.9]
    }
grid_search = GridSearchCV(m_xgb, param_grid, cv=4, scoring=SMAPE, n_jobs=-1)

grid_search_result = []
for i in tqdm(df_train1['building_type'].unique()):
    train_x = df_train1.loc[df_train1['building_type']== i, :].drop(['year', 'hour', 'building_type', 'power'], axis=1).reset_index(drop=True)
    train_y = df_train1.loc[df_train1['building_type']== i, 'power'].reset_index(drop=True)

    m_le = LabelEncoder()
    train_x['building_number'] = m_le.fit_transform(train_x['building_number'])
    
    grid_search.fit(train_x, train_y)
    result_dict = {}
    result_dict['building_type'] = i
    result_dict['n_estimators'] = grid_search.best_params_['n_estimators']
    result_dict['learning_rate'] = grid_search.best_params_['learning_rate']
    result_dict['max_depth'] = grid_search.best_params_['max_depth']
    result_dict['colsample_bytree'] = grid_search.best_params_['colsample_bytree']
    result_dict['best_score'] = grid_search.best_score_
    
    grid_search_result.append(result_dict)

bdType_best_params = pd.DataFrame(grid_search_result)


# =============================================================================
# 최종모델 학습 및 Test 데이터 predict 값 출력
# =============================================================================
building_type_list = list(df_train1['building_type'].unique())
n_estimators_list = list(bdType_best_params['n_estimators'])
learning_rate_list = list(bdType_best_params['learning_rate'])
max_depth_list = list(bdType_best_params['max_depth'])
colsample_bytree_list = list(bdType_best_params['colsample_bytree'])

answer = pd.Series()
for bd, tree, lr, md, cb in tqdm(zip(building_type_list, n_estimators_list, learning_rate_list, max_depth_list, colsample_bytree_list)):
    train_x = df_train1.loc[df_train1['building_type']== bd, :].drop(['year', 'hour', 'building_type', 'power'], axis=1).reset_index(drop=True)
    train_y = df_train1.loc[df_train1['building_type']== bd, 'power']
    test_x = df_test1.loc[df_test1['building_type']== bd, :].drop(['year', 'hour', 'building_type'], axis=1).reset_index(drop=True)
    
    m_le = LabelEncoder()
    train_x['building_number'] = m_le.fit_transform(train_x['building_number'])
    test_x['building_number'] = m_le.fit_transform(test_x['building_number'])
    
    m_xgb = XGBRegressor(n_estimators = tree, learning_rate = lr,
                         max_depth = md, colsample_bytree = cb)
    m_xgb.fit(train_x, train_y)
    pred = m_xgb.predict(test_x)
    answer = answer.append(pd.Series(pred))

answer = answer.reset_index(drop=True)
submission['answer'] = answer

submission.to_csv('submit_ver1.csv', index=False)


# =============================================================================
# =============================================================================
# 건물유형별 모델 => 점수가 좋지 않음
# 건물별 모델 생성 및 적용 (즉, 100개의 XGBRegressor 모델을 만들 것)
# =============================================================================
# =============================================================================
# =============================================================================
# 건물별 최적 독립변수 조합에 따른 최적 파라미터 값 출력
# =============================================================================
m_xgb = XGBRegressor()
param_grid = {
    'n_estimators' : [100, 500, 1000],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth' : [3, 5, 7],
    'colsample_bytree' : [0.7, 0.9]
    }
grid_search = GridSearchCV(m_xgb, param_grid, cv=4, scoring=SMAPE, n_jobs=-1)

grid_search_result = []
for i in tqdm(df_train1['building_number'].unique()):
    train_x = df_train1.loc[df_train1['building_number']== i, :].drop(['building_number', 'year', 'hour', 'building_type', 'power'], axis=1).reset_index(drop=True)
    train_y = df_train1.loc[df_train1['building_number']== i, 'power'].reset_index(drop=True)
    
    grid_search.fit(train_x, train_y)
    result_dict = {}
    result_dict['building_number'] = i
    result_dict['n_estimators'] = grid_search.best_params_['n_estimators']
    result_dict['learning_rate'] = grid_search.best_params_['learning_rate']
    result_dict['max_depth'] = grid_search.best_params_['max_depth']
    result_dict['colsample_bytree'] = grid_search.best_params_['colsample_bytree']
    result_dict['best_score'] = grid_search.best_score_
    
    grid_search_result.append(result_dict)

bdnum_best_params = pd.DataFrame(grid_search_result)


# =============================================================================
# 최종모델 학습 및 Test 데이터 predict 값 출력
# =============================================================================
building_number_list = list(df_train1['building_number'].unique())
n_estimators_list = list(bdnum_best_params['n_estimators'])
learning_rate_list = list(bdnum_best_params['learning_rate'])
max_depth_list = list(bdnum_best_params['max_depth'])
colsample_bytree_list = list(bdnum_best_params['colsample_bytree'])

answer = pd.Series()
for bd, tree, lr, md, cb in tqdm(zip(building_number_list, n_estimators_list, learning_rate_list, max_depth_list, colsample_bytree_list)):
    train_x = df_train1.loc[df_train1['building_number']== bd, :].drop(['building_number', 'year', 'hour', 'building_type', 'power'], axis=1)
    train_y = df_train1.loc[df_train1['building_number']== bd, 'power']
    test_x = df_test1.loc[df_test1['building_number']== bd, :].drop(['building_number', 'year', 'hour', 'building_type'], axis=1)
    
    m_xgb = XGBRegressor(n_estimators = tree, learning_rate = lr,
                         max_depth = md, colsample_bytree = cb)
    m_xgb.fit(train_x, train_y)
    pred = m_xgb.predict(test_x)
    answer = answer.append(pd.Series(pred))

answer = answer.reset_index(drop=True)
submission['answer'] = answer

submission.to_csv('submit_ver2.csv', index=False)
len(train_x.columns)

