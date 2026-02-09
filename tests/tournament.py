import sys
sys.path.append('.')
import torch
import rlcard
from rlcard.agents import DQNAgent
from agents.dqn import SmartDQNAgent
from rlcard.utils import tournament, Logger

from agents.xgboost_agent import XGBoostRLCardAgent as XGBoostAgent 

# 1. Configurer l'environnement à 6 joueurs
env = rlcard.make('no-limit-holdem', config={'seed': 42, 'game_num_players': 6})

# 2. Charger ton Champion (XGBoost)
# Adapte le chemin du modèle .pkl si besoin
hero_agent = XGBoostAgent(model_path='models/xgb/xgb_pluribus_2026-01-29_12-31_fe6426.json', env = env)

# 3. Charger le Challenger (DQN Buddy)
# villain_agent = DQNAgent(
#     num_actions=env.num_actions,
#     state_shape=env.state_shape[0],
#     mlp_layers=[256, [128]],
#     device='cpu'
# )
# On charge les poids qu'on vient d'entraîner
checkpoint_path = 'models/rl/dqn_smart/checkpoint_dqn.pt'

# On demande à la classe DQNAgent de se construire à partir de ce checkpoint
villain_agent = SmartDQNAgent(env,state_shape=87)


try:
    # 1. On charge le fichier (La Valise)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. On ouvre le compartiment 'q_estimator'
    if 'q_estimator' in checkpoint:
        estimator_data = checkpoint['q_estimator']
        
        # 3. On regarde si les poids sont encore cachés dans une sous-clé 'qnet'
        # (C'est souvent le cas avec RLCard)
        if 'qnet' in estimator_data:
            print("🔍 Poids trouvés dans [q_estimator][qnet].")
            network_weights = estimator_data['qnet']
        else:
            print("🔍 Poids trouvés directement dans [q_estimator].")
            network_weights = estimator_data
    else:
        network_weights = checkpoint
        
    # 4. On injecte enfin les poids dans le réseau
    villain_agent.q_estimator.qnet.load_state_dict(network_weights)
    
    # 5. Mode Évaluation
    villain_agent.q_estimator.qnet.eval()
    print("✅ SmartDQN chargé avec succès !")
    
except RuntimeError as e:
    print(f"❌ ERREUR D'ARCHITECTURE : {e}")
    print("👉 Astuce : Si ça parle de 'Missing key(s)', c'est qu'on a trouvé les poids mais que l'architecture (mlp_layers) ne correspond pas.")
    sys.exit()
except Exception as e:
    print(f"❌ Erreur critique : {e}")
    sys.exit()


agents = [hero_agent] + [villain_agent for _ in range(5)]
env.set_agents(agents)
# """
print(f"🥊 DÉBUT DU MATCH : XGBoost  vs 5 DQNs")
print("-" * 50)

result = tournament(env, 10000)

# 6. Analyse des résultats
print(result)

# CORRECTION DE L'INTERPRÉTATION
hero_payoff = result[0]        # Gain du Hero (Toi)
villain_payoff = result[1]     # Gain du premier Villain (DQN)

print("-" * 50)
print(f"📊 RÉSULTATS RÉELS SUR 1000 MAINS :")
print(f"🤖 Hero (XGBoost) : {hero_payoff:.4f} BB/Main")
print(f"😈 Villain (DQN)  : {villain_payoff:.4f} BB/Main")

print("-" * 50)
if hero_payoff > 0:
    print("✅ SUCCÈS : Ton agent gagne de l'argent !")
elif hero_payoff > -0.25:
    print("⚠️ MOYEN : Ton agent perd moins que s'il foldait tout (>-0.25).")
else:
    print("❌ ECHEC : Ton agent se fait exploiter (Pire que Fold 100%).")

    
    """
print("\n🎬 --- DÉBUT DU REPLAY --- 🎬")
state, player_id = env.reset()

# On boucle jusqu'à la fin de la main
while not env.is_over():
    
    # Récupérer les infos brutes pour l'affichage
    raw_obs = state['raw_obs']
    current_cards = raw_obs['hand']
    stage = raw_obs['stage']
    
    # Qui joue ?
    player_name = "🤖 HERO (XGB)" if player_id == 0 else f"😈 Villain {player_id}"
    
    # Si c'est le Hero, on veut voir ses cartes et stats
    if player_id == 0:
        print(f"\n🔵 {player_name} à toi ! Main: {current_cards} | Pot: {raw_obs['pot']}")
        
        # On utilise eval_step pour avoir les probas
        action, info = hero_agent.eval_step(state)
        
        # On affiche les probas de ton modèle (si dispos dans info)
        if isinstance(info, dict):
            print(f"   🧠 Cerveau: {info}")
    else:
        # Les méchants jouent
        action = env.agents[player_id].step(state)
    
    # Traduction de l'action pour l'humain
    action_str = env._decode_action(action) # ex: raise_pot
    
    print(f"   👉 {player_name} décide de : {action_str}")

    # Etape suivante
    state, player_id = env.step(action)

# Résultat final
print("\n🏁 --- FIN DE LA MAIN ---")
payoffs = env.get_payoffs()
print(f"💰 Résultat final : {payoffs}")

if payoffs[0] >= 0:
    print("✅ Le Hero n'a pas perdu d'argent !")
else:
    print(f"❌ Le Hero a perdu {payoffs[0]} jetons.")
    """

try:
    print(f"{'JOUEUR':<15} | {'GAIN':<6} | {'MAIN RÉVÉLÉE'}")
    print("-" * 45)
    
    # Hero (Toujours index 0)
    hero_hand = [str(c) for c in env.game.players[0].hand]
    print(f"🤖 HERO (XGB)    | {payoffs[0]:<+6} | {hero_hand}")
    
    # Villains
    for i in range(1, env.num_players):
        # On récupère les objets cartes et on les convertit en string
        hand_obj = env.game.players[i].hand
        hand_str = [str(c) for c in hand_obj]
        
        # Petit commentaire selon le résultat
        status = ""
        if payoffs[i] > 0: status = "🏆 WIN"
        elif payoffs[i] <= -100: status = "💀 BUST"
        
        print(f"😈 DQN {i}        | {payoffs[i]:<+6} | {hand_str} {status}")

except Exception as e:
    print(f"Impossible de révéler les mains (Accès interne bloqué): {e}")

print("-" * 45)