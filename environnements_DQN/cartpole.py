# pour créer un environnement cartpole-v1 et pour entraîner un agent DQN
import numpy as np
import gym
import stable_baselines3 as sb3
from stable_baselines3 import DQN

import matplotlib.pyplot as plt
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure,
                                                       close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True


env = gym.make('CartPole-v0')

model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./cartpole_log/")
model.learn(total_timesteps=10000, log_interval=4,
            tb_log_name="first_test", callback=TensorboardCallback())
model.save("cartpole_agent")


model = DQN.load("cartpole_agent")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
