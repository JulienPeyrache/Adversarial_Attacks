import os
import gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from fgsm import fast_gradient_method
import matplotlib.pyplot as plt


def show(model,eval_env,nb_episodes=10,nb_episodes_attaque=1,attack=False,render=False):
    episodes_rewards = []
    episode_reward = 0
    pourcentage = 0
    episodes_pourcentages = []
    obs = eval_env.reset()
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
            obs_adv = fast_gradient_method(agent_actor,torch_obs,eps_deb+eps*(k//nb_episodes_attaque),np.inf)
            #Calcuer la prédiction
            action_adv, _states_adv = model.predict(obs_adv.detach().numpy(), deterministic=True)
            if action != action_adv:
                pourcentage = (1+pourcentage*steps)/(steps+1)

        obs, reward, done, _ = eval_env.step(action)
        if render:
            eval_env.render()
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
            obs = eval_env.reset()   

    plt.subplot(211)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_rewards))],episodes_rewards)
    plt.xlabel('Puissance')
    plt.ylabel('Rewards')
    
    plt.subplot(212)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_pourcentages))],episodes_pourcentages)
    plt.xlabel('Puissance')
    plt.ylabel('Pourcentage')
    plt.show()



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
    agent.learn(total_timesteps=60000, eval_env=eval_env, reset_num_timesteps=False, eval_freq=5000, n_eval_episodes=10)
    agent.save("Agent/model")



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


show(agent,eval_env)