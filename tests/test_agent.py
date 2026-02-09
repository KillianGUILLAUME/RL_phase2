# scripts/test_agent.py

import sys
sys.path.append('.')  # Pour les imports

import rlcard
from rlcard.envs import Env
from rlcard.agents import RandomAgent
from agents.xgboost_agent import XGBoostRLCardAgent
import logging
from rlcard.envs import make

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_env():
    """
    Créer l'environnement RLCard No-Limit Hold'em
    """
    env = rlcard.make(
        'no-limit-holdem',
        config = {
            'game_num_players':6,
            'seed': None
        }
    )
    return env


def test_single_game():
    """Test basique: 1 partie"""
    logger.info("🎮 Test 1: Une partie simple\n")
    
    env = create_env()

    agent = XGBoostRLCardAgent('models/xgb/xgb_pluribus_V1.pkl', env = env)
    env.set_agents([agent] + [RandomAgent(num_actions=env.num_actions)]*5)
    # env.set_agents([agent] *6)
    trajectories, payoffs = env.run(is_training=False)
    
    logger.info(f"Résultat: {payoffs}")
    logger.info(f"XGBoost: {payoffs[0]:+.1f} chips")
    logger.info(f"Random:  {payoffs[1]:+.1f} chips")
    
    agent.print_stats()
    
    return payoffs, env, agent


def test_multiple_games(env = None, agent = None, num_games: int = 100):
    """Test sur plusieurs parties"""
    logger.info(f"🎮 Test 2: {num_games} parties\n")
    
    if env is None or agent is None:
        config = {
            'game_num_players':6,
            'seed': None
        }
        env = Env('no-limit-holdem', config)
        agent = XGBoostRLCardAgent('models/xgb/xgb_pluribus_V1.pkl')
        env.set_agents([agent] + [RandomAgent(num_actions=env.num_actions)]*5)
    
    total_payoff = [0.0, 0.0]
    wins = [0, 0]
    
    for i in range(num_games):
        _, payoffs = env.run(is_training=False)
        
        total_payoff[0] += payoffs[0]
        total_payoff[1] += payoffs[1]
        
        if payoffs[0] > payoffs[1]:
            wins[0] += 1
        elif payoffs[1] > payoffs[0]:
            wins[1] += 1
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Parties {i+1}/{num_games}...")
    
    logger.info(f"\n📊 Résultats sur {num_games} parties:")
    logger.info(f"  XGBoost: {total_payoff[0]/num_games:+.2f} chips/game | {wins[0]} victoires ({wins[0]/num_games*100:.1f}%)")
    logger.info(f"  Random:  {total_payoff[1]/num_games:+.2f} chips/game | {wins[1]} victoires ({wins[1]/num_games*100:.1f}%)")
    
    agent.print_stats()
    
    return env, agent


# def test_fullring():
    """Teste l'agent en partie complète (6 joueurs)"""
    from rlcard.envs import make
    # === IMPORTANT: 6 joueurs au lieu de 2 ===
    env = make('no-limit-holdem', config={'game_num_players': 6})
    
    agent = XGBoostRLCardAgent(
        model_path='models/xgb/xgb_pluribus_V1.pkl',
        env=env
    )
    
    # Remplacer le joueur 0 par ton agent
    env.set_agents([agent]  * 6)
    
    fold_situations = 0
    fold_taken = 0
    total_decisions=0
    
    for hand_num in range(100):
        trajectories, payoffs = env.run(is_training=False)
        print(len(trajectories[0]))
        # Analyser les décisions du joueur 0
        for transition in trajectories[0]:
            total_decisions += 1
            print(transition.keys())
            state = transition['state']
            action = transition['action']
            
            legal_actions = [a.value for a in state.get('raw_legal_actions', [])]
            
            # Compter les situations où FOLD était possible
            if 0 in legal_actions:
                fold_situations += 1
                
                if action == 0:
                    fold_taken += 1
                    raw_obs = state.get('raw_obs', {})
                    print(f"✅ FOLD main #{hand_num}")
                    print(f"   Hand: {raw_obs.get('hand', '?')}")
                    print(f"   Stage: {raw_obs.get('stage', '?')}")
                    print(f"   Pot: {raw_obs.get('pot', '?')}")
    
    agent.print_stats()
    
    print(f"\n{'='*60}")
    print(f"📊 ANALYSE FOLD (Full Ring - 6 joueurs)")
    print(f"{'='*60}")
    print(f"Décisions totales: {total_decisions}")
    print(f"Situations où FOLD était légal: {fold_situations}")
    print(f"FOLDs pris: {fold_taken} ({100*fold_taken/max(fold_situations,1):.1f}%)")
    print(f"{'='*60}\n")
def test_fullring():
    """Teste avec traçage en temps réel"""
    
    
    # === IMPORTANT: 6 joueurs au lieu de 2 ===
    env = make('no-limit-holdem', config={'game_num_players': 6})
    
    # Créer un agent wrapper qui trace ses décisions
    base_agent = XGBoostRLCardAgent(
        model_path='models/xgb/xgb_pluribus_V1.pkl',
        env=env
    )

    
    # Remplacer le joueur 0 par ton agent
    # env.set_agents([agent]  * 6)
    
    # Stats
    stats = {
        'fold_situations': 0,
        'fold_taken': 0,
        'call_situations': 0,
        'call_taken': 0,
        'raise_situations': 0,
        'raise_taken': 0,
        'total': 0
    }
    
    class TrackedAgent:
        def __init__(self, base_agent, stats):
            self.base_agent = base_agent
            self.stats = stats
        
        def eval_step(self, state):
            action, probs = self.base_agent.eval_step(state)
            
            self.stats['total'] += 1
            
            legal_actions = [a.value for a in state.get('raw_legal_actions', [])]
            
            if 0 in legal_actions:
                self.stats['fold_situations'] += 1
                if action == 0:
                    self.stats['fold_taken'] += 1
                    print(f"✅ FOLD! (proba={probs.get(0, 0):.3f})")
            
            if 1 in legal_actions:
                self.stats['call_situations'] += 1
                if action == 1:
                    self.stats['call_taken'] += 1
            
            if 2 in legal_actions:
                self.stats['raise_situations'] += 1
                if action == 2:
                    self.stats['raise_taken'] += 1
            
            return action, probs
        
        def step(self, state):
            return self.base_agent.step(state)
        
        @property
        def use_raw(self):
            return self.base_agent.use_raw
    
    tracked_agent = TrackedAgent(base_agent, stats)
    env.set_agents([tracked_agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(5)])
    
    print("\n🎮 Lancement de 100 mains...\n")
    
    for hand_num in range(100):
        trajectories, payoffs = env.run(is_training=False)
        if (hand_num + 1) % 20 == 0:
            print(f"  Main {hand_num + 1}/100 terminée...")
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Décisions totales: {stats['total']}")
    print(f"\n🔴 FOLD: {stats['fold_taken']}/{stats['fold_situations']} ({100*stats['fold_taken']/max(stats['fold_situations'],1):.1f}%)")
    print(f"🟡 CALL: {stats['call_taken']}/{stats['call_situations']} ({100*stats['call_taken']/max(stats['call_situations'],1):.1f}%)")
    print(f"🟢 RAISE: {stats['raise_taken']}/{stats['raise_situations']} ({100*stats['raise_taken']/max(stats['raise_situations'],1):.1f}%)")
    print(f"{'='*60}\n")


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == 'fullring':
#         test_fullring()
#     elif len(sys.argv) > 1 and sys.argv[1] == 'realtime':
#         test_fullring_realtime()
#     else:
#         test_single_game()

def debug_legal_actions():
    """Découvre les vraies valeurs des actions légales"""
    
    env = make('no-limit-holdem', config={'game_num_players': 6})
    
    agent = XGBoostRLCardAgent(
        model_path='models/xgb/xgb_pluribus_V1.pkl',
        env=env
    )
    
    env.set_agents([agent] + [RandomAgent(num_actions=env.num_actions) for _ in range(5)])
    
    print("\n🔍 ANALYSE DES ACTIONS LÉGALES\n")
    
    all_legal_values = set()
    all_legal_names = set()
    
    for hand_num in range(20):  # 20 mains suffisent
        trajectories, payoffs = env.run(is_training=False)
        
        for transition in trajectories[0]:
            raw_legal_actions = transition.get('raw_legal_actions', [])
            
            print(f"\n📍 Décision #{hand_num}:")
            print(f"   Type de raw_legal_actions: {type(raw_legal_actions)}")
            print(f"   Contenu: {raw_legal_actions}")
            
            for action in raw_legal_actions:
                print(f"\n   Action:")
                print(f"      Type: {type(action)}")
                print(f"      Valeur: {action}")
                
                # Si c'est un enum
                if hasattr(action, 'value'):
                    print(f"      .value: {action.value}")
                    all_legal_values.add(action.value)
                
                # Si c'est un enum avec name
                if hasattr(action, 'name'):
                    print(f"      .name: {action.name}")
                    all_legal_names.add(action.name)
                
                # Essayer d'autres attributs
                if hasattr(action, '__dict__'):
                    print(f"      __dict__: {action.__dict__}")
            
            # Regarder aussi 'legal_actions' (pas 'raw_legal_actions')
            legal_actions = transition.get('legal_actions', [])
            print(f"\n   legal_actions (pas raw): {legal_actions}")
            print(f"   Type: {type(legal_actions)}")
            
            if hand_num >= 2:  # 3 décisions suffisent pour voir le pattern
                break
        
        if hand_num >= 2:
            break
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    print(f"Valeurs trouvées (.value): {sorted(all_legal_values)}")
    print(f"Noms trouvés (.name): {sorted(all_legal_names)}")
    print("="*60)


# if __name__ == "__main__":
#     import sys
    
#     if len(sys.argv) > 1 and sys.argv[1] == 'debug-actions':
#         debug_legal_actions()
#     elif len(sys.argv) > 1 and sys.argv[1] == 'fullring':
#         test_fullring()
#     elif len(sys.argv) > 1 and sys.argv[1] == 'realtime':
#         test_fullring()
#     else:
#         test_single_game()


if __name__ == "__main__":
    # debug_legal_actions()
    # test_fullring()
    try:
        # Test progressif
        logger.info("🚀 Début des tests\n")
        _, env, agent = test_single_game()
        n=10000
        input(f"\n⏸️  Appuie sur Enter pour lancer {n} parties...")
        env_multi, agents = test_multiple_games(env, agent, n)
    
        # ✅ Accéder à l'agent
        logger.info("\n✅ Tests terminés !")
        env_multi.agents[0].print_stats()
        
    except KeyboardInterrupt:
        logger.info("\n⏸️  Tests interrompus")
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}", exc_info=True)