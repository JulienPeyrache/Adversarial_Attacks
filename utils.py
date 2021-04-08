import torch
import matplotlib.pyplot as plt
import numpy as np

def show(agent,eval_env,nb_episodes=10,nb_episodes_attaque=1,attack=False,model_fn=None,attack_function=None,render=False,ecart=0.5):
    episodes_rewards = []
    episode_reward = 0
    pourcentage = 0
    episodes_pourcentages = []
    obs = eval_env.reset()
    eps_deb = 0
    eps_fin = 0.3
    k = 0
    steps = 0
    while k<nb_episodes*nb_episodes_attaque:
        action,_ = agent.predict(obs)
        if attack:
            if nb_episodes == 1:
                eps = eps_fin
            else:
                eps = (eps_fin-eps_deb)/(nb_episodes-1)
            #Convertir en un tenseur
            torch_obs = torch.Tensor(np.expand_dims(obs,axis=0))
            #Calculer l'image adversielle
            obs_adv = attack_function(model_fn,torch_obs,eps_deb+eps*(k//nb_episodes_attaque),ecart=ecart)
            #Calcuer la prédiction
            action_adv,_ = agent.predict(obs_adv.detach())
            if action != action_adv:
                pourcentage = (1+pourcentage*steps)/(steps+1)
            obs, reward, done, _ = eval_env.step(action_adv[0])
        else:
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