from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

texts = ['You are the Best',
         'You are the Nice']

tokenizer = Tokenizer(num_words = 10, oov_token = '<OOV>')
tokenizer.fit_on_texts(texts)

#텍스트 데이터를 정수 인덱스 형태로 변환
sequences = tokenizer.texts_to_sequences(texts)

#이진 형태로 인코딩
binary_results = tokenizer.sequences_to_matrix(sequences, mode = 'binary')

print(tokenizer.word_index)
print('---------------------')

print(f'sequences: {sequences}\n')
print(f'binary_vectors:\n {binary_results}\n')

#원-핫 형태로 인코딩
#print(to_categofical(sequences))

test_text = ['You are the One']
test_seq = tokenizer.texts_to_sequences(test_text)

print(f'test sequences: {test_seq}')
