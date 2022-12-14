import gym
import numpy as np
import torch
from utils.fgsm_attacks import fgsm_classification

import stable_baselines3
from stable_baselines3 import DQN
import matplotlib.pyplot as plt


def show(model,nb_episodes=10,nb_episodes_attaque=1,attack=False,render=False):
    episodes_rewards = []
    episode_reward = 0
    pourcentage = 0
    episodes_pourcentages = []
    obs = env.reset()
    eps_deb = 0
    eps_fin = 0.4
    k = 0
    steps = 0
    while k<nb_episodes*nb_episodes_attaque:
        action, _states = model.predict(obs, deterministic=True)
        if attack:
            if nb_episodes == 1:
                eps = eps_fin
            else:
                eps = (eps_fin-eps_deb)/(nb_episodes-1)
            #Convertir en un tenseur
            torch_obs = torch.Tensor(np.expand_dims(obs,axis=0))
            #Calculer l'image adversielle
            obs_adv = fgsm_classification(model.q_net,torch_obs,eps_deb+eps*(k//nb_episodes_attaque),np.inf)
            #Calcuer la prédiction
            action_adv, _states_adv = model.predict(obs_adv.detach().numpy(), deterministic=True)
            if action != action_adv:
                pourcentage = (1+pourcentage*steps)/(steps+1)

        obs, reward, done, _ = env.step(action_adv[0])
        if render:
            env.render()
        episode_reward += reward
        steps += 1
        if done:
            k += 1
            if k%nb_episodes_attaque==0:
                steps = 0
                episodes_pourcentages.append(pourcentage)
                pourcentage = 0
                episodes_rewards.append(episode_reward/nb_episodes_attaque)
                episode_reward = 0
            obs = env.reset()   

    plt.subplot(211)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_rewards))],episodes_rewards)
    plt.xlabel('Puissance')
    plt.ylabel('Rewards')
    
    plt.subplot(212)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_pourcentages))],episodes_pourcentages)
    plt.xlabel('Puissance')
    plt.ylabel('Pourcentage')
    plt.show()



env = gym.make("CartPole-v0")
eval_env = gym.make("CartPole-v0")
env.reset()
eval_env.reset()
#env = stable_baselines3.common.monitor.Monitor(env)
policy_kwargs = dict(net_arch=[32,16])
try:
    model = DQN.load("DQN1",env)
    print("Model loaded")
except:
    model = DQN("MlpPolicy", env, tensorboard_log="Agent/tensorboard", policy_kwargs=policy_kwargs, exploration_final_eps=0.05,learning_rate=0.001)
    print("Could not load the model")
    model.learn(total_timesteps=200000,eval_env=eval_env,reset_num_timesteps=False,eval_freq=10000,n_eval_episodes=10)
    model.save("DQN1")




#episodes_rewards = env.get_episode_rewards()
#plt.plot([i for i in range(len(episodes_rewards))],episodes_rewards)
#plt.show()

show(model,attack=True,nb_episodes_attaque=5)