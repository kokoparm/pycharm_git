import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('./models/book_data_max_6772_size_214146.npy', allow_pickle=True)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
# embedding = 벡터라이징
model.add(Embedding(214146, 100, input_length=6772)) # embedding(단어의 개수, 차원)
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(LSTM(128, activation='tanh', return_sequences=True)) # 문자열 해당, return...=True 다른 lstm을 더 사용하기 위해
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(23, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=100, epochs=8, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test)
# print(score[1])
model.save('./models/book_model.h5'.format(score[1]))