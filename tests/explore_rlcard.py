"""
Découvrir comment fonctionne RLCard.
Ce fichier est juste pour comprendre, pas pour produire.
"""

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, tournament
import numpy as np

print("=" * 60)
print("🃏 EXPLORATION DE RLCARD")
print("=" * 60)

# Initialiser l'environnement
set_seed(42)
env = rlcard.make('no-limit-holdem', config={'seed': 42})


print("--- DÉCODAGE DES ACTIONS ---")
for action_id in range(env.num_actions):
    try:
        decoded = env.decode_action(action_id)
        print(f"Action {action_id} correspond à : {decoded}")
    except:
        pass

# Ajouter 2 agents aléatoires
env.set_agents([
    RandomAgent(num_actions=env.num_actions),
    RandomAgent(num_actions=env.num_actions)
])

print(f"\nNombre d'actions possibles: {env.num_actions}")
print(f"Forme de l'état: {env.state_shape}")
print(f"Actions: {env.actions}")

# Jouer une main et voir ce qui se passe
print("\n--- SIMULATION D'UNE MAIN ---")
trajectories, payoffs = env.run(is_training=False)

print(f"\n✅ Résultat: Joueur 0 = {payoffs[0]}, Joueur 1 = {payoffs[1]}")

# Explorer la structure des trajectoires
print("\n--- STRUCTURE DES TRAJECTOIRES ---")
print(f"Type de trajectories: {type(trajectories)}")
print(f"Nombre de joueurs: {len(trajectories)}")
print(f"Type de trajectories[0]: {type(trajectories[0])}")
print(f"Longueur de trajectories[0]: {len(trajectories[0])}")

# Première décision du joueur 0
print("\n--- PREMIÈRE DÉCISION DU JOUEUR 0 ---")
first_decision = trajectories[0][0]

print(f"Type de first_decision: {type(first_decision)}")

# Vérifier si c'est un dictionnaire
if isinstance(first_decision, dict):
    print(f"Clés disponibles: {list(first_decision.keys())}")
    print(f"\nÉtat reçu (shape): {np.array(first_decision['obs']).shape}")
    print(f"État reçu (5 premières valeurs): {np.array(first_decision['obs'])[:5]}")
    print(f"Actions légales: {first_decision['legal_actions']}")
else:
    print(f"⚠️ Ce n'est pas un dictionnaire, c'est: {first_decision}")

# Analyser TOUS les éléments de la trajectoire
print("\n--- ANALYSE COMPLÈTE DE LA TRAJECTOIRE DU JOUEUR 0 ---")
for i, element in enumerate(trajectories[0]):
    print(f"\n📍 Élément {i+1}/{len(trajectories[0])}")
    print(f"   Type: {type(element)}")
    
    if isinstance(element, dict):
        print(f"   Clés: {list(element.keys())}")
        
        # Observer l'état
        if 'raw_obs' in element:
            obs = element['raw_obs']
            if isinstance(obs, np.ndarray):
                if obs.ndim > 0:
                    print(f"   État: array de taille {len(obs)}")
                else:
                    print(f"   État: scalaire numpy = {obs.item()}")
            else:
                print(f"   État: {obs}")
        
        # Observer les actions légales
        if 'raw_legal_actions' in element:
            print(f"   Actions légales: {list(element['raw_legal_actions'])}")
    
    elif isinstance(element, (int, float, np.number)):
        print(f"   Valeur scalaire: {element}")
    
    elif isinstance(element, np.ndarray):
        print(f"   Array de shape: {element.shape}")
    
    else:
        print(f"   Contenu: {element}")

# Simuler 1000 mains
print("\n" + "=" * 60)
print("--- TOURNOI DE 1000 MAINS ---")
payoffs_total = tournament(env, 1000)
print(f"Résultats moyens sur 1000 mains:")
print(f"  Joueur 0 (Random): {payoffs_total[0]:+.2f}")
print(f"  Joueur 1 (Random): {payoffs_total[1]:+.2f}")

print("\n" + "=" * 60)
print("✅ Exploration terminée !")
print("\n📚 Ce qu'on a appris:")
print("1. RLCard gère TOUT le jeu (règles, pot, cartes, etc.)")
print("2. Les trajectoires ont une structure particulière")
print("3. On reçoit toujours les actions légales quand nécessaire")
print("4. Les agents Random sont équilibrés (moyenne ≈ 0)")
print("\n🎯 Prochaine étape:")
print("   → Créer un agent XGBoost qui bat les Random !")
print("=" * 60)
