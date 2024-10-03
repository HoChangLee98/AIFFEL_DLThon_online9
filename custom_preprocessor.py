import pandas as pd
import json
import os
## 정규식 
import re
## 형태소 분석기
from konlpy.tag import Okt
# ## 띄어쓰기 교정
# from konlpy.tag import Hannanum


class Preprocessor:
    def __init__(
        self, 
        folder_path="../data/"
    ):
        self.folder_path = folder_path
        self.train = pd.read_csv(self.folder_path + "train.csv")
        train_general_text = pd.read_csv("train_general.csv")
        train_general = train_general_text.loc[3950:, ["idx", "conversation"]]
        train_general["class"] = "일반 대화"
        
        self.train = pd.concat([self.train, train_general])
        
        with open (self.folder_path + "/test.json", "r") as f:
            test = json.load(f)
        self.test = pd.DataFrame(test).T.reset_index().rename(columns={"index":"file_name"})
        
        f = open("../stopword.txt", "r")
        self.stopword_list = f.read().splitlines()
    

    def preprocess(
        self, 
        version:int=0
    ):
        """실제로 전처리를 진행하는 함수
        
        Args: 
            version: 버전 입력
        Returns:
            preprocessed_train: 전처리가 끝난 훈련 데이터 셋 
            preprocessed_test: 전처리가 끝난 테스트 데이터 셋 
        """
        train_file_path = "../preprocessed_data/" + f"preprocessed_train_v{version}.csv"
        test_file_path = "../preprocessed_data/" + f"preprocessed_test_v{version}.csv"
        
        ## train data
        if not os.path.exists(train_file_path):
            ## 원본 데이터 복사
            preprocessed_train = self.train.copy()

            print("train 데이터 셋을 생성합니다.")
            preprocessed_train["conversation"] = preprocessed_train["conversation"].apply(lambda x: self.text_cleaner(x))

            print("train 데이터 셋을 저장합니다.")
            preprocessed_train.to_csv(train_file_path, index=False)
        
        else:
            preprocessed_train = pd.read_csv(train_file_path)
        
        ## test data 
        if not os.path.exists(test_file_path):
            ## 원본 데이터 복사
            preprocessed_test = self.test.copy()

            print("test 데이터 셋을 생성합니다.")
            preprocessed_test["text"] = preprocessed_test["text"].apply(lambda x: self.text_cleaner(x))

            print("test 데이터 셋을 저장합니다.")
            preprocessed_test.to_csv(test_file_path, index=False)        
        
        else:
            preprocessed_test = pd.read_csv(test_file_path)
            
        
        return preprocessed_train, preprocessed_test
        
    def text_cleaner(self, text):
        """text 데이터를 전처리하는 함수 

        Args:
            text: 하나의 텍스트 데이터 샘플
        Returns:
            cleaned_text: 전처리된 하나의 텍스트 데이터 샘플
        """        
        ## <s>, [INST], </s>, [/INST] 제거
        text = text.replace("<s>", "").replace("[INST]", "").replace("</s>", "").replace("[/INST]", "")
        
        ## 정규식 전처리: 한글, 영어, 숫자를 제외하고 모두 제거한다.  
        text = re.sub("[^가-힣a-zA-Z0-9]+", " ", text)
        
        ## 여러 개의 공백을 하나의 공백으로
        text = re.sub(r'\s+', ' ', text).strip()

        ## 형태소 분석기 사용
        okt = Okt()
        tokens = okt.morphs(text)

        ## 한국어 불용어 처리
        tokens = [word for word in tokens if not word in self.stopword_list]
        cleaned_text = " ".join(tokens)    

        return cleaned_text      