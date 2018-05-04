## Lecturer: Sung Kim

## Cost function과 Optimizer 설명 참고
## http://bcho.tistory.com/1139

'''
## Cost function
(원래값과 계산된 값의 차이) = 측정값 - 그래프의 값
d = y_data - y_origin

ABS(d1) + ABS(d2) + ... + ABS(dn) / n 이 최소가 되는 W, b의 값을 구하는 것
:: 최소자승법
이를 연산하는 함수를 Cost function이라 한다.

## Optimizer
위 cost function의 최소값을 찾는 알고리즘을 옵티마이져(Optimizer)라고 한다
Optimizer의 한 종류인 경사 하강법(Gradient Descent)을 여기서는 사용

예) Cost가 y축이고 W가 x축일 때, 그래프 모양이 y=x^2을 보인다고 하면, W에 대한 적정값에 대한 예측을 시작하는 점을
기준으로 경사가 아래로 되어 있는 부분으로 점을 움직이며 미분을 통해 기울기가 0이 되는 점을 찾는다

위의 지식을 바탕으로,
1. 학습: Cost function을 정의하고 실제 데이터 x_data, y_data들을 넣어서 경사하강법에 의해 Cost가 최소가 되는 W, b를 구한다
- 이 작업은 W값을 변화시키면서 반복적으로 x_data로 계산하여, 실제 측정 데이터와 가설에 의해서 예측된 결과값에 대한 차이를 찾아내어 W, b를 구한다

2. 예측: 학습된 모델을 바탕으로 새로운 데이터를 넣는다 
'''

import tensorflow as tf

## H(x) = Wx + b ##

## Variable: Tensorflow가 학습하는 과정에서 변형을 하는 변수: Trainable variable
## 여기서 W, b는 cost function의 최소값을 찾기 위해 반복적으로 변해야 한다
W = tf.Variable(tf.random_normal([1]), name = 'weight') ## tf.random_normal: 몇차원 array인가?
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])

## 가설 세우고
hypothesis = W * X + b

## cost function 정의한다. cost(W,b): 최소자승법 식 
cost = tf.reduce_mean(tf.square(hypothesis - Y))

## 경사 하강법 optimizer 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
## train 노드를 실행
train = optimizer.minimize(cost)

## Launch the graph in a session
sess = tf.Session()
## To use W, b, we need to initializes global variables in the graph
sess.run(tf.global_variables_initializer())

## Fit the line
## 2001번 learning 시킬 것이고 20번째마다 출력
## feed_dict를 통해 학습 시킬 (실제)데이터를 준다
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
                                feed_dict = {X:[1,2,3], Y:[1,2,3]})
    
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

