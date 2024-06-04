from tensorflow.keras.datasets.mnist import load_data

#텐서플로우세엇 데이터를 다운받기
(x_train, y_train), (x_test, y_test) = load_data(path= 'mnist.npz')
#학습데이터
print(x_train.shape, y_train.shape)
print(y_train)

#테스트 데이터
print(x_test.shape, y_test.shape)
print(y_test)

import matplotlib.pyplot as plt
import numpy as np

sample_size = 3
#0~59999의 범위에서 무작위로 세 개의 정수를 뽑기
random_idx = np.random.randint(6000, size=sample_size)

for idx in random_idx:
  img = x_train[idx, :]
  label = y_train[idx]
  plt.figure()
  plt.imshow(img)
  plt.title('%d-th data, label is %d' % (idx,label))
  
from sklearn.model_selection import train_test_split

#훈련/테스트 데이터를  0.7/ 0.3의 비율로 분리
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state = 777)

print(f'훈련데이터 {x_train.shape} 레이블 {y_train.shape}')
print(f'검증데이터 {x_val.shape} 레이블 {y_val.shape}')

num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]

#모델의 입력으로 사용하기 위한 전처리 과정
x_train = (x_train.reshape((num_x_train, 28 * 28))) / 255
x_val = (x_val.reshape((num_x_val, 28 * 28))) / 255
x_test = (x_test.reshape((num_x_test, 28 * 28))) / 255

print(x_train.shape)

from tensorflow.keras.utils import to_categorical

#각 데이터의 레이블을 범주형 형태로 변경합니다.
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(y_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
#입력데이터의 형태를 꼭 명시
#784차원의 데이터를 입력 받고, 64개의 출력을 가지는 첫번째 Dense 층
model.add(Dense(64, activation = 'relu', input_shape = (784, )))
model.add(Dense(32, activation = 'relu')) #32개의 출력을 가지는 Dense층
model.add(Dense(10, activation = 'softmax')) #10개의 출력을 가지는 신경망

def softmax(arr):
  m = np.max(arr)
  arr = arr - m #exp의 오버플로우 방지
  arr = np.exp(arr)
  return arr / np.sum(arr)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

case_1 = np.array([3.1, 3.0, 2.9])
case_2 = np.array([2.0, 1.0, 0.7])

np.set_printoptions(precision=3) #numphy 소수점 제한
print(f'sigmoid {sigmoid(case_1)}, softmax {softmax(case_1)}')
print(f'sigmoid {sigmoid(case_2)}, softmax {softmax(case_2)}')

#옵티마이저 : Adam 손실함수 : categorical_crossentropy, 모니터링 할 평가지표 : acc
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs = 30, batch_size = 128, validation_data = (x_val, y_val))

results = model.predict(x_train)
model.evaluate(x_test, y_test)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize= (7,7))
cm = confusion_matrix(np.argmax(y_test, axis = -1)), np.argmax(results, axis=)
sns.heatmap(cm, annot= True, fmt= 'd', cmap= 'Blues')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()