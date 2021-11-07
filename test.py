import gym

env = gym.make('CartPole-v0')

for i_episode in range(20):
  observation = env.reset()                   # First observation

  for t in range(100):                        # For 100 time steps
      env.render()
      print(observation)
      action = env.action_space.sample()      # Take a random action
      observation, reward, done, info = env.step(action)

      if done:                                # Finish the episode if done
          print('Episode finished after {} timesteps'.format(t+1))
          break
env.close()