import os
import gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

try:
    os.mkdir("PPO_Agent")
except:
    print("Dir already exist")
train_env = gym.make('LunarLanderContinuous-v2')
train_env.reset()

eval_env = gym.make('LunarLanderContinuous-v2')
eval_env.reset()

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[dict(vf=[64,64], pi=[64,64])])
try:
    agent = PPO.load("PPO_Agent/model",train_env)
    print("loaded")
except:
    print("could not load")
    agent = PPO(MlpPolicy, train_env, learning_rate=3e-4, policy_kwargs=policy_kwargs, tensorboard_log="PPO_Agent/tensorboard")

agent.learn(total_timesteps=100000, eval_env=eval_env, reset_num_timesteps=False, eval_freq=5000, n_eval_episodes=10)

agent.save("PPO_Agent/model")



#FGSM
def agent_actor(obs):
    latent_pi, _, _ = agent.policy._get_latent(obs)
    action_logits = agent.policy.action_net(latent_pi)
    action_distribution = F.softmax(action_logits,dim=1)
    return action_distribution

#FGSM Regression
def agent_critic(obs):
    _, latent_vf, _ = agent.policy._get_latent(obs)
    value = agent.policy.value_net(latent_vf)
    return value




def agent_act(obs):
    action, value, log_prob = agent.policy(obs)
    return action




