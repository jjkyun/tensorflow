'''
Loading data using tensorflow

Lecturer: Sung Kim &
http://bcho.tistory.com/1163


텐서플로우에서 모델을 학습 시킬 때, 학습 데이터를 모델에 적용하는 방법으로 feeding이라는 방법을 사용
그러나 학습 데이터가 크면 이를 메모리에 모두 탑재할 수 없기 때문에 파일에서 읽어드리며 학습 해야한다

Tensorflow Queue를 사용!
Queue Runner: Queue에 데이터를 넣는다 (Enqueue)
- Enqueue_operation: 어떻게 queue에 데이터를 넣을지 정의
- 멀티 쓰레드로 작동하는데, 이 쓰레드들을 관리해주기 위해 별도로 Coordinator를 사용

추가 기능: Reader와 Decoder
1. 파일 목록을 읽는다.
2. 읽은 파일목록을 filename queue에 저장한다.
3. Reader 가 finename queue 에서 파일명을 하나씩 읽어온다.
4. Decoder에서 해당 파일을 열어서 데이타를 읽어들인다.
5. 필요하면 읽어드린 데이타를 텐서플로우 모델에 맞게 정재한다. (이미지를 리사이즈 하거나, 칼라 사진을 흑백으로 바꾸거나 하는 등의 작업)
6. 텐서 플로우에 맞게 정재된 학습 데이타를 학습 데이타 큐인 Example Queue에 저장한다.
7. 모델에서 Example Queue로 부터 학습 데이타를 읽어서 학습을 한다.
'''

import numpy as np
import tensorflow as tf

## 단순히 하나의 데이터를 loading 할 때 
xy = np.loadtxt('filename.csv', delimiter=',', dtype=np.float32)
'''
Example of data in file
# EXAM1, EXAM2, EXAM3, FINAL
73, 80, 75, 152
93, 88, 93, 185
89, 91, 90, 180
'''

## split data
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

## filename queue를 만들고 file들을 저장
filename_queue = tf.train.string_input_producer(
    ['filename.csv', 'filename1.csv', '...'], shuffle = False, name = 'filename_queue')

## Reader를 통해 Queue에 있는 text file을 읽어온다
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

## 들고온 value 값을 어떤 식으로 parsing 할 것인지 설정
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults = record_defaults)

## 읽어드린 값들을 batch로 묶어서 반환
## batch로 묶고자 하는 tensor 들을 인자로 준 다음에, batch_size (한번에 묶어서 리턴하고자 하는 텐서들의 개수)를 정해주면 된다
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

sess = tf.Session()

## Building coordinator that manage queue multi thread
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

coord.request_stop()
coord.join(threads)
