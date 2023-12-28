# import gym

# env = gym.make('ALE/Breakout-v5', full_action_space=False, obs_type="grayscale")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42)

# print(observation.shape, type(observation), type(env.reset(seed=42)))


# # for _ in range(1000):
# #     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

# #     if terminated or truncated:
# #         observation, info = env.reset()

# # env.close()

from game_env import TrainingEnv

trainer = TrainingEnv('ALE/Breakout-v5', history_len=4)

trainer.train()