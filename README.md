# DACON 2023 전력사용량 예측 AI 경진대회

Git-hub repository:
https://github.com/DTDisScience/project.git

직전 대회 우승팀인 j_sean 팀의 코드를 참고하여 진행하였습니다.

data set
- Train data: csv\train.csv
- building info data: csv\building_info.csv
- test data: csv\teset.csv
- submission(제출용): csv\sample_submission.csv


# Contents
1. [프로젝트 개요](#section1)

2. [데이터 전처리](#section2)
   1. [데이터셋 설명](#sec2p1)
   2. [EDA](#sec2p2)
   3. [독립변수 추가](#sec2p3)
   4. [평가지표 SMAPE](#sec2p4)

3. [모델 생성](#section3)
   1. [건물유형별 모델](#sec3p1)
   2. [건물별 모델](#sec3p2)

4. [결론](#section4)


## 1. 프로젝트 개요 <a name="section1"></a>
 - 안정적이고 효율적인 에너지 공급을 위하여 전력 사용량 예측 시뮬레이션을 통한 효율적인 인공지능 알고리즘 발굴을 목표로 한국에너지공단에서 개최한 대회 참여
 - 관련기사
   - 필요 전력량 급증하자 꺼낸 '새 원전 건설'...상황은 첩첩산중
   - https://m.hankookilbo.com/News/Read/A2023071115570002238
 - 프로젝트를 진행하기에 앞서 팀원들과 건물별 모델, 건물유형별 모델 중 어떠한 방식으로 진행할 것인지 논의
 - 건물별 모델
   - train 데이터가 각 건물별 2040개 밖에 되지 않아, 모델의 성능을 장담할 수 없을 것으로 예상
   - 또한, 주관사인 한국에너지공단에서 새로운 건물데이터 추가될 시 예측이 어려워질 것으로 예상
 - 건물유형별 모델
   - 건물유형에 따라 일별, 시간대별 평균 전력소비량에 유사한 패턴을 보이는 것으로 확인


## 2. 데이터 전처리 <a name="section2"></a>
### 2.1 데이터셋 설명 <a name="sec2p1"></a>
1) Train data
   - 독립변수
     - 건물별 기상 data (건물별 2040개, 총 204000개 data)
     - 측정기준: 2022년 6월 1일 00시 ~ 2022년 8월 24일 23시 (1시간 단위)
     - 변수: 건물번호, 일시, 기온, 강수량, 풍속, 습도, 일조, 일사, 전력소비량
   - 종속변수
     - 건물별 1시간 단위 전력사용량
2) Test data
   - 독립변수
     - 건물별 기상 Data (건물별 168개, 총 16800개 data)
     - 측정기준: 2022년 8월 25일 00시 ~ 2022년 8월 31일 23시 (1시간 단위)
     - 변수: 건물번호, 일시, 기온, 강수량, 풍속, 습도, 일조, 일사, 전력소비량
3) Building_info
   - 변수 : 건물번호, 건물유형, 연면적, 냉방면적, 태양광용량, ESS저장용량, PCS용량

### 2.2 EDA <a name="sec2p2"></a>
1) 불필요 컬럼 제거, 결측치 확인 및 처리
   - 세부내용 final.py 파일을 참고
3) 건물유형별 일별 전력소비량 패턴 시각화
 - 건물유형별 일별 평균 전력소비량
![12Mean Power Consumption by Day and Building Type](https://github.com/DTDisScience/project/assets/131218231/b9ecb621-2c0f-45b6-a0c5-bbc34bf474a9)
 - 건물유형별 각 건물의 시간대별 전력소비량
   ![UNIVERSITY Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/f6a3677f-ea7a-4f30-a8a4-2395eac0ccc3)
   ![PUBLIC Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/58aa3ded-ff3f-4d47-ab7d-dcc568161615)
   ![OTHER BUILDINGS Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/a320dc57-e596-4a12-9423-8627fc4bee5c)
   ![LABORATORY Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/31af938d-a57f-441b-a8b0-7613180a86b4)
   ![KNOWLEDGE INDUSTRY CENTER Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/24670ac6-ef0c-407b-a84a-258c5e3b8c32)
   ![HOTEL AND RESORT Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/3205b41c-b2d3-49b3-b76a-af2c9635e8fd)
   ![HOSPITAL Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/90df23ed-8a2c-4c60-a9ed-17fa5de9cf56)
   ![DISCOUNT MART Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/0ec5ee64-a555-4932-8b5a-49512994b027)
   ![DEPARTMENT STORE AND OUTLET Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/55ea842c-2896-4410-810b-903752666f9a)
   ![DATA CENTER Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/4c1af1c5-9b8b-43a2-9c3a-34aff5e6356f)
   ![COMMERCIAL Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/a3c2669f-5f3c-4aff-abf3-11191947bc6e)
   ![APARTMENT Mean Power Consumption by Hour](https://github.com/DTDisScience/project/assets/131218231/083cfcff-cecc-441c-bc42-bdac7729bbbd)

건물유형별로 일별 전력소비량이 유사한 패턴을 보임을 확인할 수 있었습니다.<br/> 
따라서 우선, 건물유형별로 전력소비량 예측 모델을 만들었습니다.

### 2.3 독립변수 추가 <a name="sec2p3"></a>
1) 건물유형별, 월별, 시간대별 전력사용량 평균, 표준편차, 최대사용량<br/>
   (단, 모든 건물이 6월 대비 7,8월 전력사용량이 급증하므로 월별로 산출)
2) 각 건물별, 월별, 요일별, 앞뒤 세시간 평균, 최대 전력사용량
3) 불쾌지수(TDI)
4) 온도, 습도 각 시간대별 변화량

### 2.4 평가지표 SMAPE <a name="sec2p4"></a>
- 대회의 심사기준인 SMAPE(Symmetric Mean Absolute Percentage Error)를 사용
  ![그림1](https://github.com/DTDisScience/project/assets/131218231/c014a7e5-6cfb-40de-a045-c4a14a529526)


## 3. 모델 생성 <a name="section3"></a>
시계열 도메인에서 주로 사용하는 부스팅 기반 모델인 xgboost 모델 적용<br/>
실제로 시계열 데이터에서 우수한 성능을 보이는 LSTM(Long Short Term Memory) 보다 높은 점수를 기록했습니다.<br/> 
시계열 데이터를 회귀 데이터로 변경했기 때문에, 변수가 시계열 특성을 반영할 수 있도록 가공

### 3.1 건물유형별 모델 <a name="sec3p1"></a>
건물유형별로 모델의 최적 Parameter를 찾고 test 데이터 예측값을 출력하였습니다
- 건물유형별 최적 Parameter
  ![건물유형별_param](https://github.com/DTDisScience/project/assets/131218231/66ebd602-1976-44f5-b11d-eb69c7f972e7)

### 3.2 건물별 모델 <a name="sec3p2"></a>
건물별로 모델의 최적 Parameter를 찾고 test 데이터 예측값을 출력하였습니다
- 건물별 최적 Parameter
  ![건물별_param](https://github.com/DTDisScience/project/assets/131218231/ce74daf4-bb47-4c13-9698-7f7b8c8ce9ac)


## 4. 결론 <a name="section4"></a>
- 전체 1233 팀 중 136위 (상위 약 11%)
![순위](https://github.com/DTDisScience/project/assets/131218231/775b67ab-3a6b-4932-b1d8-3863cfe88cbd)

1. EDA의 중요성
   - 건물유형별 일별 소비패턴에서 상이한 패턴을 보이는 건물을 별도로 처리했어야 한다는 아쉬움이 있습니다.
   - 할인마트의 경우 그 면적으로 볼 때, 월 2회 휴무를 진행하며,<br/> 
     train 데이터를 통해 실제로 월 2회 전력사용량이 급감하는 것을 확인할 수 있었습니다.<br/> 
     다만, 2022년 8월의 경우 광복절 대체휴무일로 인하여 test 데이터 중 1일 휴무 여부를 확인할 수 없었습니다.
2. 종속변수의 분포 확인
   - 다른 참여자들의 코드를 통해, 종속변수인 전력소비량이 좌측편포를 보임을 알 수 있었습니다.<br/> 
     종속변수의 로그변환을 통해 정규분포와 가까운 형태로 변환하여<br/>
     보다 나은 예측결과를 도출할 수 있을 것으로 예상합니다.
3. 최적 Parameter 관련
   - SMAPE 점수에 따른 Parameter 선택이 아닌<br/> 
     각 Parameter가 높거나 낮다는 것이 정확히 어떠한 의미를 가지며,<br/> 
     어떠한 값이 모델의 과적합을 방지할 수 있는지에 대한 저의 부족함을 느낄 수 있었습니다.
