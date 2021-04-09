import gym
import torch
import torch.nn.functional as F
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt

import fgsm

def show(agent,eval_env,nb_episodes=10,nb_episodes_attaque=1,attack=False,model_fn=None,attack_function=None,render=False,targeted=False,y=None):
    episodes_rewards = []
    episode_reward = 0
    pourcentage = 0
    episodes_pourcentages = []
    prob_action = 0
    episodes_prob_action = []
    val_state = 0
    episodes_val_state = []
    obs = eval_env.reset()
    eps_deb = 0
    eps_fin = 0.3
    k = 0
    steps = 0
    while k<nb_episodes*nb_episodes_attaque:
        #Convertir en un tenseur
        torch_obs = torch.Tensor(np.expand_dims(obs,axis=0))
        action,_ = agent.predict(obs)
        if attack:
            if nb_episodes == 1:
                eps = eps_fin
            else:
                eps = (eps_fin-eps_deb)/(nb_episodes-1)
            #Calculer l'image adversielle
            obs_adv = attack_function(model_fn,torch_obs,eps_deb+eps*(k//nb_episodes_attaque),norm=np.inf,targeted=targeted,y=y)
            #Calcuer la prédiction
            action_adv,_ = agent.predict(obs_adv.detach())
            obs, reward, done, _ = eval_env.step(action_adv[0])
            if action != action_adv:
                pourcentage = (1+pourcentage*steps)/(steps+1)

        else:
            obs, reward, done, _ = eval_env.step(action)
        if model_fn == agent_actor:
            prob_action = (((torch.max(agent_actor(torch_obs)-agent_actor(obs_adv)))+prob_action*steps)/(steps+1))
        if model_fn == agent_critic:
            val_state = (((torch.max(agent_critic(torch_obs)-agent_critic(obs_adv)))+val_state*steps)/(steps+1))
        if render:
            eval_env.render()
        episode_reward += reward
        steps += 1
        if done:
            k += 1
            if k%nb_episodes_attaque==0:
                steps=0
                episodes_rewards.append(episode_reward/nb_episodes_attaque)
                episode_reward = 0
                episodes_pourcentages.append(pourcentage)
                pourcentage = 0
                episodes_prob_action.append(prob_action)
                prob_action = 0
                episodes_val_state.append(val_state)
                val_state = 0
            obs = eval_env.reset()   

    plt.subplot(221)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_rewards))],episodes_rewards,marker='x')
    plt.xlabel('Puissance')
    plt.ylabel('Rewards')
    
    plt.subplot(222)
    plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_pourcentages))],episodes_pourcentages,marker='d')
    plt.xlabel('Puissance')
    plt.ylabel('Pourcentage')
    plt.legend(['fgsm_classification_untargeted','fgsm_classification_targeted','fgsm_regression_untargeted','fgsm_regression_targeted'])
    if model_fn == agent_actor:
        plt.subplot(223)
        plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_prob_action))],episodes_prob_action,marker='o')
        plt.xlabel('Puissance')
        plt.ylabel('prob_action')
    if model_fn == agent_critic:
        plt.subplot(224)
        plt.plot([eps_deb+i*(eps_fin-eps_deb)/(nb_episodes-1) for i in range(len(episodes_val_state))],episodes_val_state,marker='*')
        plt.xlabel('Puissance')
        plt.ylabel('val_state')


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



if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = PPO.load("PPO_Agent/model")
    plt.xlabel('Puissance')
    plt.ylabel('Rewards')
    show(agent,env,nb_episodes_attaque=2,attack=True,model_fn=agent_actor,attack_function=fgsm.fast_gradient_method,targeted=False)
    show(agent,env,nb_episodes_attaque=2,attack=True,model_fn=agent_actor,attack_function=fgsm.fast_gradient_method,targeted=True,y=torch.tensor([1]))
    
    show(agent,env,nb_episodes_attaque=2,attack=True,model_fn=agent_critic,attack_function=fgsm.fast_gradient_method_regression,targeted=False)
    show(agent,env,nb_episodes_attaque=2,attack=True,model_fn=agent_critic,attack_function=fgsm.fast_gradient_method_regression,targeted=True,y=torch.tensor([1,0],dtype=torch.float))
    
    plt.show()



"""
obs = env.reset()
for i in range(50):

    torch_obs = torch.Tensor(np.expand_dims(obs,axis=0))
    print("original observation : {}".format(obs))
    print("policy on original observation : {}".format(agent_actor(torch_obs)))
    print("value on original observation : {}".format(agent_critic(torch_obs)))
    action = agent_act(torch_obs)
    print("action taken on original observation: {}".format(action.item()))
    print()

    #adv_obs = fgsm.fast_gradient_method(agent_actor, torch_obs,eps=0.2,norm=2, targeted=False).detach().numpy()
    adv_obs = fgsm.fast_gradient_method_regression(agent_critic, torch_obs, eps=0.2,norm= 2, targeted=True).detach().numpy()
    
    torch_adv_obs = torch.Tensor(np.expand_dims(adv_obs,axis=0))
    print("perturbed observation : {}".format(adv_obs))
    print("policy on perturbed observation : {}".format(agent_actor(torch_adv_obs)))
    print("value on perturbed observation : {}".format(agent_critic(torch_adv_obs)))
    action = agent_act(torch_adv_obs)
    print("action taken on perturbed observation: {}".format(action.item()))
    print()
    print()
    print()
    print()

    obs, reward, done, info = env.step(action.item())
    
    env.render()
    
    if done:
      obs = env.reset()
"""