# Installation de l'environnement virtuel

- Création un environement virtuel à la racine du projet avec la commande : `python3 -m venv <chemin d'accès au dossier du projet>\venv` sur Windows ou `python3 - m venv <chemin d'accès au dossier du projet>/venv sur macOS`
- Activation de l'environnement virtuel avec la commande : `<chemin d'accès au dossier du projet>\venv\Scripts\activate.bat` sur Windows ou `source <chemin d'accès au dossier du projet>/venv/bin/activate` sur macOS. "(venv)" devrait apparaitre devant la ligne de commande.
- Insallation des packages nécessaires dans l'environnement virtuel avec la commande: `pip install -r requirements.txt`
- Déactivation de l'environnement virtuel avec la commande : `deactivate`

# Installation de l'environnement virtuel

- Création d'un environement virtuel conda : `conda create -n attaques-adverses python=3.6.12`
- Activation de l'environnement virtuel avec la commande : `conda activate attaques-adverses`. "(attaques adverses)" devrait apparaitre devant la ligne de commande.
- Insallation des packages nécessaires dans l'environnement virtuel avec la commande: `pip install -r requirements.txt`
- Déactivation de l'environnement virtuel avec la commande : `conda deactivate`

# Maintenance du requirements.txt

Si le code ne fonctionne pas car il manque un package, signalez-le au groupe et ajoutez le package dans _requirements.txt_

# Structure du code

Les tests de code, qui nous impriment les résultas, se situent à la racine du projet. Nous avons également un dossier pour créer les environnements DQN et entraîner un agent DQN et un dernier dossier contenant les différentes attaques adverses

# Comment lancer les tests

## Lancer fgsm_adversial

Il faut lancer le code fgsm_adversarial pour obtenir le résultat
(fgsm correspond à l'implémentation de l'algorithme
et simpleCNN coresspond à la création d'un réseau de convolution)

Nécessite tensorflow et Numpy
