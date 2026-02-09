import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import print_card
from agents.xgboost_agent import XGBoostRLCardAgent as XGBoostAgent
# import torch
import sys

from agents.human_console import HumanConsoleAgent


# ==========================================
# LE JEU
# ==========================================

def play_game():
    # 1. Setup
    env = rlcard.make('no-limit-holdem', config={'seed': 42, 'game_num_players': 6})
    
    # 2. Chargement du Boss (Ton XGBoost)
    print("🤖 Chargement du Boss XGBoost...")
    boss = XGBoostAgent('models/xgb/xgb_pluribus_V1.json', env=env)
    print('ici')
    
    # 3. Chargement des Figurants (DQN ou Random)
    # On met des Randoms pour remplir la table, ou tes DQNs si tu veux que ce soit dur
    # Ici je mets des Randoms pour que tu te concentres sur le duel contre le Boss
    fillers = [RandomAgent(env.num_actions) for _ in range(4)]
    
    # 4. L'Humain
    human = HumanConsoleAgent(env.num_actions)
    
    # Ordre : [Boss, Fillers..., Humain]
    # Tu seras le Joueur 5 (dernier de parole, la meilleure position !)
    agents = [boss] + fillers + [human]
    env.set_agents(agents)

    print("\n🎰 DÉBUT DE LA PARTIE (Humain vs XGBoost)")
    print("Tu es le Joueur 5 (Dernier). Le Boss est le Joueur 0.")
    
    # Boucle de jeu infinie
    while True:
        print("\n" + "♠♥♦♣"*10)
        print("NOUVELLE MAIN")
        print("♠♥♦♣"*10)
        
        trajectories, payoffs = env.run(is_training=False)
        
        # Fin de la main
        human_payoff = payoffs[5]
        boss_payoff = payoffs[0]
        
        print("\n🏁 RÉSULTAT DE LA MAIN :")
        print(f"🤖 Boss (J0) : {boss_payoff}")
        print(f"👤 Toi  (J5) : {human_payoff}")
        
        if human_payoff > 0:
            print("🎉 Bravo, tu as gagné !")
        elif human_payoff < 0:
            print("💀 Tu as perdu des jetons...")
            
        match input("\nAppuie sur [Entrée] pour rejouer ou 'q' pour quitter..."):
            case 'q': break
            case _: continue

if __name__ == "__main__":
    play_game()