import sys
sys.path.append('.')

import numpy as np

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from agents.dqn import SmartDQNAgent
from rlcard.utils import set_seed, tournament, reorganize, Logger
import torch

from agents.xgboost_agent import XGBoostRLCardAgent as XGBoostAgent 


# 1. Configurer l'environnement
env = rlcard.make('no-limit-holdem', config={'seed': 42})
hero_agent = XGBoostAgent(model_path='models/xgb/xgb_pluribus_V1.pkl', env = env)

def evaluate(agent, env, n_episodes=100):
    """Évalue l'agent contre des adversaires aléatoires."""
    agents = [agent] + [RandomAgent(env.num_actions) for _ in range(env.num_players - 1)]
    env.set_agents(agents)
    rewards = []
    for _ in range(n_episodes):
        trajectories, payoffs = env.run(is_training=False)
        rewards.append(payoffs[0])
    return np.mean(rewards)


smart = True

if smart:
    agent = SmartDQNAgent(env, model_path='models/rl/dqn_smart')
else:
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=torch.device('cpu'),
        
        # Paramètres d'apprentissage ajustés
        replay_memory_size=50000,      # Mémoire plus grande pour se souvenir des erreurs passées
        batch_size=32,
        replay_memory_init_size=500, # On le laisse observer un peu plus avant d'apprendre
        update_target_estimator_every=100,
        epsilon_decay_steps=20000 # Il explore longtemps avant de se figer
    )


agents = [agent] + [hero_agent for _ in range(5)]
env.set_agents(agents)

print("🏋️‍♂️ Début de l'entraînement du Sparring Partner DQN...")


for episode in range(1000):

    if episode % 1000 == 0:
        avg_reward = evaluate(agent, env)
        print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")
    
    trajectories, payoffs = env.run(is_training=True)

    trajectories = reorganize(trajectories, payoffs)
    
    for ts in trajectories[0]:
        agent.feed(ts)

    if episode % 250 == 0:
        print(f"   Episode {episode}/10000 terminé...")

print("✅ Entraînement terminé !")

# 5. Sauvegarder ce modèle pour ne pas le ré-entraîner à chaque fois
import os
# save_path = 'models/rl/dqn_buddy'
# os.makedirs(save_path, exist_ok=True)
# agent.save_checkpoint(save_path)
# print(f"💾 Agent DQN sauvegardé dans {save_path}")

# 5. Sauvegarde Finale
# RLCard sauvegarde souvent automatiquement si save_path est défini, mais on force ici
# save_dir = 'models/rl/dqn_smart'
# os.makedirs(save_dir, exist_ok=True)
# # Astuce : DQNAgent a une méthode save_checkpoint, mais selon la version ça change.
# # Le plus sûr est de sauvegarder l'état interne si la méthode n'existe pas.
# try:
#     agent.save_checkpoint(save_dir)
#     print(f"💾 Agent sauvegardé dans {save_dir}")
# except AttributeError:
#     # Fallback si méthode non trouvée (dépend version RLCard)
#     torch.save(agent.q_estimator.qnet.state_dict(), os.path.join(save_dir, 'checkpoint_dqn.pt'))
#     print(f"💾 Poids du réseau sauvegardés manuellement dans {save_dir}