# DACON 2023 전력사용량 예측 AI 경진대회

Git-hub repository:
https://github.com/DTDisScience/project.git

직전 대회 우승팀인 j_sean 팀의 코드를 참고하여 진행하였습니다.

data set
- Train data : csv\train.csv
- building info data : csv\building_info.csv
- test data : csv\teset.csv
- submission(제출용) : csv\sample_submission.csv


# Contents
1. [프로젝트 개요](#section1)

2. [데이터 전처리](#section2)
   1. [데이터셋 설명](#sec2p1)
   2. [EDA](#sec2p2)
   3. [독립변수 추가](#sec2p3)
   4. [평가지표 SMAPE](#sec2p4)

4. [모델 생성](#section3)
   1. [건물유형별 모델](#sec3p1)
   2. [개별 건물별 모델](#sec3p2)

5. [결론](#section4)


## 1. 프로젝트 개요 <a name="project outline"></a>
 - 안정적이고 효율적인 에너지 공급을 위하여 전력 사용량 예측 시뮬레이션을 통한 효율적인 인공지능 알고리즘 발굴을 목표로 한국에너지공단에서 개최한 대회 참여
 - 관련기사
   - 필요 전력량 급증하자 꺼낸 '새 원전 건설'...상황은 첩첩산중
   - https://m.hankookilbo.com/News/Read/A2023071115570002238
 - 프로젝트를 진행하기에 앞서 팀원들과 건물별 모델, 건물유형별 모델 중 어떠한 방식으로 진행할 것인지 논의
 - 건물별 모델
   - train 데이터가 각 모델별로 2040개 밖에 되지 않아, 모델의 성능을 장담할 수 없을 것으로 예상
   - 또한, 주관사인 한국에너지공단에서 새로운 건물데이터 추가될 시 예측이 어려워질 것으로 예상
 - 건물유형별 모델
   - 건물유형에 따라 일별, 시간대별 평균 전력소비량에 유사한 패턴을 보이는 것으로 확인


## 2. 데이터 전처리 <a name="section2"></a>
### 2.1 데이터셋 설명 <a name="sec2p1"></a>
1. Train data
   - 독립변수
     - 건물별 기상 Data (건물별 2040개, 총 204000개 data)
     - 측정기준: 2022년 6월 1일 00시 ~ 2022년 8월 24일 23시 (1시간 단위)
     - 변수: 건물번호, 일시, 기온, 강수량, 풍속, 습도, 일조, 일사, 전력소비량
   - 종속변수
     - 건물별 1시간 단위 전력사용량
2. Test data
   - 독립변수
     - 건물별 기상 Data (건물별 168개, 총 16800개 data)
     - 측정기준: 2022년 8월 25일 00시 ~ 2022년 8월 31일 23시 (1시간 단위)
     - 변수: 건물번호, 일시, 기온, 강수량, 풍속, 습도, 일조, 일사, 전력소비량
3. Building_info
   - 변수 : 건물번호, 건물유형, 연면적, 냉방면적, 태양광용량, ESS저장용량, PCS용량

### 2.2 EDA <a name="sec2p2"></a>
1. 불필요 컬럼 제거 및 결측치 확인 및 처리
2. 건물유형별 일별 전력소비량 패턴 시각화
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




















