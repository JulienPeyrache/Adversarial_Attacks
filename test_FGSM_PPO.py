import os
import gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from utils.fgsm_attacks import fgsm_classification, fgsm_regression
import numpy as np
from utils.show_results import show



#FGSM
def agent_actor(obs):
    latent_pi, _, _ = agent.policy._get_latent(obs)
    action_logits = agent.policy.action_net(latent_pi)
    action_distribution = torch.nn.functional.softmax(action_logits,dim = 1)
    return action_distribution

#FGSM Regression
def agent_critic(obs):
    _, latent_vf, _ = agent.policy._get_latent(obs)
    value = agent.policy.value_net(latent_vf)
    return value

def agent_act(obs):
    action, _, _ = agent.policy(obs)
    return action



if __name__ == '__main__':
    try:
        os.mkdir("Agent")
    except:
        print("Dossier Agent existe deja")

    train_env = gym.make('CartPole-v1')
    train_env.reset()

    eval_env = gym.make('CartPole-v1')
    eval_env.reset()


    try:
        agent = PPO.load("Agent/model",train_env)
        print("Model loaded")
    except:
        print("Could not load the model")
        agent = PPO(MlpPolicy, train_env, tensorboard_log="Agent/tensorboard")
        agent.learn(total_timesteps=70000, eval_env=eval_env, reset_num_timesteps=False, eval_freq=5000, n_eval_episodes=10)
        agent.save("Agent/model")

    # Show agent results without attack
    show(agent,eval_env,attack=False, nb_episodes_attaque=2,render=False, plot_title="No attack")
    # Show agent results with classification attack
    show(agent,eval_env,attack=True,model_fn=agent_actor,attack_function=fgsm_classification,nb_episodes_attaque=2, render= False,plot_title="Classification attack")
    # Show agent results with regression attack
    show(agent,eval_env,attack=True,model_fn=agent_critic,attack_function=fgsm_regression,nb_episodes_attaque=2, render=False, plot_titile="Regression attack")