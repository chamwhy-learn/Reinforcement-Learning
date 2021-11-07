import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 뉴럴 네트워크 모델 만들기
model = keras.models.Sequential([
    keras.layers.Dense(24, input_dim=4, activation=tf.nn.relu),
    keras.layers.Dense(24, activation=tf.nn.relu),
    keras.layers.Dense(2, activation='linear')
])

# 모델 컴파일
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error')

score = []
memory = deque(maxlen=2000)

# CartPole 환경 구성
env = gym.make('CartPole-v0')

# 100회의 에피소드 시작
for i in range(100):

    state = env.reset()
    state = np.reshape(state, [1, 4])
    eps = 1 / (i / 50 + 10)

    # 200 timesteps
    for t in range(200):
        env.render()
        # Inference: e-greedy
        if np.random.rand() < eps:
            action = np.random.randint(0, 2)
        else:
            predict = model.predict(state)
            action = np.argmax(predict)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done or t == 199:
            print('Episode', i, 'Score', t + 1)
            score.append(t + 1)
            break

    # Training
    if i > 10:
        minibatch = random.sample(memory, 16)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.9 * np.amax(model.predict(next_state)[0])
            target_outputs = model.predict(state)
            target_outputs[0][action] = target
            print("model fit")
            model.fit(state, target_outputs, epochs=1, verbose=0)

env.close()
print(score)
plt.plot(range(len(score)), score)
plt.show()
