# AIFFEL_DLThon_online9

### 초기 설정
아래 코드를 lms cloud shell에서 실행해주세요!

```bash
cd aiffel
git clone https://github.com/HoChangLee98/dlthon_team5.git
```

### 데이터 불러오기
---
기존의 데이터 셋에 허깅 페이스의 일반 대화 1000개 데이터를 합침  

### 데이터 전처리 
---
1. 불러온 데이터 셋에 포함된 `<s>`, `[INST]`, `</s>`, `[/INST]` 제거
2. 정규식 전처리
3. 여러 개의 공백을 하나의 공백으로 변환
4. 형태소 단위로 토큰화 후 불용어 처리 

### 사용 모델 
---  
| 모델                                | 검증 정확도 결과   | 실제 f1-score       | 학습 방법                  | 비고                          |
|-------------------------------------|-------------------|--------------------|---------------------------|-------------------------------|
| LSTM                                | F1-score: 0.096   | -                  | -                        | 정확도가 낮고 시간이 오래 걸림   |
| Transformer                         | F1-score: 0.8668  | F1-score: 0.2499   | 직접 구현                 | transformer의 인코더 부분만 사용|
| GPT-api 활용                        | -                 | F1-score: 0.5178   | api 사용                  | 프롬프트 학습을 통해 추론        |
| TFBertForSequenceClassification v0  | F1-score: 0.9890  | F1-score: 0.5100   | pretrained에 전체 학습    | pretrained 모델을 불러와 학습   |
| TFBertForSequenceClassification v1  | F1-score:         | -                  | pretrained에 분류기만 학습 | pretrained 모델을 불러와 학습   |
| KoBERT                              | Accuracy: 0.9047  | F1-score: 0.6293   | pretrained에 전체 학습    | memory 부족으로 코랩에서 진행    |
