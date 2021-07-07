'''

전처리 과정
1. concat / 완료
2. null, 중복 값 제거 / 완료
3. 행태소 분리, 불용어 처리
4. 데이터셋 생성

'''

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
import pickle, re, os
import warnings
warnings.filterwarnings(action='ignore')

if not 'claen_data(cate10).csv' in os.listdir('./crawling_csv/'):
    df = pd.read_csv('./crawling_csv/claened_data(cate10).csv')
    X = df['Introduction']

    for i in range(len(X)):
        X[i] = re.sub('[^가-힣|a-z|A-Z|0-9]', ' ', X[i])
    df.to_csv('./crawling_csv/claen_data(cate10).csv')
    print('clean_data.csv 생성이 완료되었습니다.')
    print('실행 내용 : 파일 컴파일, [Introduction](''가-힣 | a-z | A-Z | 0-9'')')

df = pd.read_csv('./crawling_csv/claen_data(cate10).csv')
X = df['Introduction']  # 학습 시킬 데이터
Y = df['Small_category']  # 라벨

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_
with open('./models/label_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)
onehot_Y = to_categorical(labeled_Y)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i])
    if (i % 20 == 0) and (i > 1):
        print('.', end='')
    if (i % 200 == 0) and (i > 1):
        print('{} / {}'.format(i, len(X)))

stopwords = pd.read_csv('../read_file/stopwords.csv')
for i in range(len(X)):
  result = []
  for j in range(len(X[i])):
    if len(X[i][j]) > 1:
      if X[i][j] not in list(stopwords['stopword']):
        result.append(X[i][j])
  X[i] = ' '.join(result)
  if (i % 20 == 0) and (i>1):
    print('.', end='')
  if i % 200 == 0:
    print('{} / {}'.format(i, len(X)))

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
with open('./models/words_token.pickle', 'wb') as f:
    pickle.dump(token, f)
wordsize = len(token.word_index) + 1
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])

X_pad = pad_sequences(tokened_X, max)
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
xy = X_train, X_test, Y_train, Y_test
np.save('./models/book_data_max_{}_size_{}.npy'.format(max, wordsize), xy)



