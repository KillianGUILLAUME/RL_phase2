"""
🥊 Tournament : faire jouer différents modèles les uns contre les autres.

Usage:
    python tests/tournament.py --hero <chemin_modèle> --villain <chemin_modèle> [--num_games 10000] [--num_villains 5]

Formats supportés:
    .json / .pkl  → XGBoost
    .zip          → PPO (MaskablePPO / SB3) — compatible 87 et 91 features
    .pt           → DQN (SmartDQNAgent)
    random        → Agent aléatoire
"""

import sys
sys.path.append('.')
import os
import argparse
import torch
import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import tournament

from agents.xgboost_agent import XGBoostRLCardAgent
from agents.dqn import SmartDQNAgent


# ═══════════════════════════════════════════════════════════
# AUTO-DÉTECTION ET CHARGEMENT DE MODÈLES
# ═══════════════════════════════════════════════════════════

def load_agent(model_path: str, env, label: str = "Agent") -> object:
    """
    Charge automatiquement un agent selon l'extension du fichier.
    
    Args:
        model_path: Chemin vers le modèle ou 'random'
        env: Environnement RLCard
        label: Nom affiché pour le logging
    
    Returns:
        Agent compatible RLCard (avec .step() et .eval_step())
    """
    if model_path == 'random':
        print(f"  🎲 {label}: Agent Aléatoire")
        return RandomAgent(num_actions=env.num_actions)
    
    if model_path == 'rule':
        from agents.rule_agents import RuleBasedBot
        agent = RuleBasedBot(env=env)
        print(f"  📏 {label}: RuleBasedBot V1")
        return agent
    
    if model_path == 'rule_v2':
        from agents.rule_agents import AdvancedRuleBot
        agent = AdvancedRuleBot(env=env)
        print(f"  🎯 {label}: AdvancedRuleBot V2")
        return agent
    
    if not os.path.exists(model_path):
        print(f"  ❌ {label}: Fichier introuvable → {model_path}")
        print(f"     Fallback → Agent Aléatoire")
        return RandomAgent(num_actions=env.num_actions)
    
    ext = os.path.splitext(model_path)[1].lower()
    basename = os.path.basename(model_path)
    
    if ext in ('.json', '.pkl'):
        # XGBoost
        agent = XGBoostRLCardAgent(model_path=model_path, env=env)
        print(f"  🌲 {label}: XGBoost → {basename}")
        return agent
    
    elif ext == '.zip':
        # PPO (SB3) — PPOBotAgent gère la compatibilité 87/91 features
        from training.sb3wrapper import PPOBotAgent
        agent = PPOBotAgent(model_path=model_path, env=env)
        dim = agent.model_input_dim
        print(f"  🧠 {label}: PPO ({dim} features) → {basename}")
        return agent
    
    elif ext == '.pt':
        # DQN
        agent = SmartDQNAgent(env, state_shape=99)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'q_estimator' in checkpoint:
            estimator_data = checkpoint['q_estimator']
            weights = estimator_data.get('qnet', estimator_data)
        else:
            weights = checkpoint
        
        agent.q_estimator.qnet.eval()
        print(f"  ⚡ {label}: DQN → {basename}")
        return agent
    
    else:
        print(f"  ❓ {label}: Extension inconnue '{ext}' → Fallback Aléatoire")
        return RandomAgent(num_actions=env.num_actions)


# ═══════════════════════════════════════════════════════════
# TOURNAMENT
# ═══════════════════════════════════════════════════════════

def run_tournament(hero_path: str, villain_paths: list, num_games: int = 10000):
    """Lance un tournoi Hero vs [Villain 1, Villain 2...]."""
    
    num_players = len(villain_paths) + 1
    env = rlcard.make('no-limit-holdem', config={'game_num_players': num_players})
    
    print("=" * 60)
    print("🥊 CONFIGURATION DU TOURNOI")
    print("=" * 60)
    print(f"  📊 {num_games} mains | {num_players} joueurs\n")
    
    # Charger les agents
    hero = load_agent(hero_path, env, label="HERO")
    villains = [load_agent(vp, env, label=f"VILLAIN_{i+1}") for i, vp in enumerate(villain_paths)]
    
    # Construire la table : [Hero, Villain 1, Villain 2, ...]
    agents = [hero] + villains
    env.set_agents(agents)
    
    print(f"\n{'─' * 60}")
    print(f"🎰 Lancement du tournoi rotatif ({num_games} mains)...")
    print(f"{'─' * 60}\n")
    
    # --- Custom Tournament Loop with Seat Rotation ---
    raw_payoffs = [0.0 for _ in range(num_players)]
    
    for game_idx in range(num_games):
        # On décale les agents d'un cran vers la droite à chaque partie (Rotation du Bouton)
        shift = game_idx % num_players
        rotated_agents = agents[-shift:] + agents[:-shift]
        env.set_agents(rotated_agents)
        
        # Jouer la main
        _, payoffs = env.run(is_training=False)
        
        # On remet les gains dans l'ordre initial [Hero, V1, V2...]
        for seat_idx, p in enumerate(payoffs):
            original_idx = (seat_idx - shift) % num_players
            raw_payoffs[original_idx] += p
    
    # Calcul des moyennes
    result = [p / num_games for p in raw_payoffs]
    # ---------------------------------------------
    hero_payoff = result[0]
    villain_avg = np.mean([result[i] for i in range(1, num_players)])
    
    print(f"📊 RÉSULTATS ({num_games} mains)")
    print(f"{'─' * 60}")
    print(f"  🤖 HERO    : {hero_payoff:+.4f} BB/main")
    print(f"  😈 VILLAIN : {villain_avg:+.4f} BB/main (moyenne)")
    
    for i in range(num_players):
        if i == 0:
            label, emoji = "HERO", "🤖"
        else:
            basename = os.path.basename(villain_paths[i-1]) if villain_paths[i-1] not in ['rule', 'rule_v1', 'rule_v2', 'random'] else villain_paths[i-1]
            label, emoji = f"V{i} ({basename})", "😈"
        print(f"     {emoji} Siège {i} [{label[:25]:<25}]: {result[i]:+.4f} BB/main")
    
    print(f"{'─' * 60}")
    if hero_payoff > 0.1:
        print("🏆 VICTOIRE NETTE : Hero domine !")
    elif hero_payoff > 0:
        print("✅ VICTOIRE : Hero gagne.")
    elif hero_payoff > -0.25:
        print("⚠️ MOYEN : Hero perd mais mieux que fold-all (> -0.25).")
    else:
        print("❌ DÉFAITE : Hero se fait exploiter.")
    
    return result


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="🥊 Tournoi de poker : Hero vs Villain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # XGBoost vs PPO (self-play)
  python tests/tournament.py \\
    --hero models/xgb_87_features/xgb_pluribus_2026-01-29_12-31_fe6426.json \\
    --villain models/rl_87_features/ppo_self_play/ppo_sb3_512x512_poker_1000000_steps.zip

  # XGBoost vs PPO (entraîné contre XGBoost)
  python tests/tournament.py \\
    --hero models/xgb_87_features/xgb_pluribus_2026-01-29_12-31_fe6426.json \\
    --villain models/rl_87_features/PPO_Poker_xgb_1770802762/FINAL_MODEL_end_steps.zip

  # XGBoost vs Random (baseline)
  python tests/tournament.py --hero models/xgb_87_features/xgb_pluribus_2026-01-29_12-31_fe6426.json --villain random

  # PPO vs PPO (self-play eval)
  python tests/tournament.py \\
    --hero models/rl_87_features/PPO_Poker_xgb_1770802762/FINAL_MODEL_end_steps.zip \\
    --villain models/rl_87_features/ppo_self_play/ppo_sb3_poker_500000_steps.zip
        """
    )
    
    parser.add_argument('--hero', '-H', type=str, required=True,
                        help="Chemin du modèle Hero (.json/.pkl/.zip/.pt ou 'random')")
    parser.add_argument('--villains', '-V', type=str, nargs='+', required=True,
                        help="Chemins des modèles Villains (séparés par un espace)")
    parser.add_argument('--num_games', '-n', type=int, default=10000,
                        help="Nombre de mains à jouer (défaut: 10000)")
    
    args = parser.parse_args()
    
    run_tournament(
        hero_path=args.hero,
        villain_paths=args.villains,
        num_games=args.num_games
    )