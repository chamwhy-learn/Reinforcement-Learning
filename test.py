import gym

env = gym.make('Acrobot-v1')

score = []

for i in range(1000):
    state = env.reset()
    done = False
    t=0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        t+=1
    print('Episode', i+1, 'Score', t + 1)
    score.append(t + 1)

env.close()
