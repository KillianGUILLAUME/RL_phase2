# tests/test_feature_extractor_full.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from core.game_state import GameState
from features.feature_builder import FeatureExtractor
from adapters.rlcard_adapter import RLCardAdapter
import rlcard
from rlcard.agents import RandomAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rlcard_integration():
    """Test 1: Intégration avec RLCard"""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: Intégration RLCard → GameState → Features")
    print("=" * 70)
    
    # Setup
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 3})
    extractor = FeatureExtractor()
    
    # Collecte de quelques observations
    observations = []
    
    for game_idx in range(5):
        env.reset()
        
        for step in range(20):  # Max 20 steps par partie
            if env.is_over():
                break
            
            player_id = env.get_player_id()
            obs = env.get_state(player_id)
            
            # Conversion en GameState
            try:
                game_state = RLCardAdapter.to_game_state(obs, env)
                
                # Extraction des features
                features = extractor.extract(game_state)
                
                observations.append({
                    'game_id': game_idx,
                    'step': step,
                    'player_id': player_id,
                    'street': game_state.street,
                    'position': game_state.position,
                    'hole_cards': str(game_state.hole_cards),
                    'board': str(game_state.board),
                    'stack_bb': game_state.effective_stack_bb,
                    'pot_bb': game_state.pot_size_bb,
                    'features': features
                })
                
            except Exception as e:
                logger.error(f"Erreur à game {game_idx}, step {step}: {e}")
                continue
            
            # Action aléatoire
            state = env.get_state(env.get_player_id())
            legal_actions_dict = state.get('legal_actions', {})
            if isinstance(legal_actions_dict, dict):
                legal_actions = list(legal_actions_dict.keys())
            else:
                legal_actions = legal_actions_dict
            action = np.random.choice(legal_actions) if legal_actions else 0
            env.step(action)
            observations.append(state)
    
    print(f"\n✅ {len(observations)} observations collectées")
    
    # Analyse des features
    # if observations:
    #     all_features = np.array([obs['features'] for obs in observations])
        
    #     print("\n📊 Statistiques des features:")
    #     print(f"  Shape: {all_features.shape}")
    #     print(f"  Min: {all_features.min():.3f}")
    #     print(f"  Max: {all_features.max():.3f}")
    #     print(f"  Mean: {all_features.mean():.3f}")
    #     print(f"  Std: {all_features.std():.3f}")
        
    #     # Vérification des NaN
    #     nan_count = np.isnan(all_features).sum()
    #     print(f"  NaN count: {nan_count}")
        
    #     if nan_count > 0:
    #         print("  ⚠️ WARNING: NaN détectés !")
    #         nan_features = np.where(np.isnan(all_features).any(axis=0))[0]
    #         feature_names = extractor.get_feature_names()
    #         print("  Features avec NaN:")
    #         for idx in nan_features:
    #             print(f"    - {feature_names[idx]} (index {idx})")
        
    #     # Distribution des features
    #     print("\n📊 Distribution de quelques features clés:")
    #     feature_names = extractor.get_feature_names()
        
    #     key_features = [
    #         'equity', 'spr_normalized', 'pot_odds', 
    #         'aggression_factor', 'position_normalized'
    #     ]
        
    #     for fname in key_features:
    #         if fname in feature_names:
    #             idx = feature_names.index(fname)
    #             values = all_features[:, idx]
    #             print(f"  {fname:25s}: min={values.min():.3f}, "
    #                   f"mean={values.mean():.3f}, max={values.max():.3f}")
    
    for i, obs in enumerate(observations[:3]):
        try:
            # Conversion RLCard → GameState
            game_state = RLCardAdapter.to_game_state(obs, env)
            # print('ici\n\n\n')
            # print(game_state)
            # print(f'\ngame state type : {type(game_state)}')
            
            # Extraction features | game_state
            features = extractor.extract(game_state)
            
            print(f"\n  Observation {i+1}:")
            print(f"    Street: {game_state.street}")
            print(f"    Position: {game_state.position}")
            print(f"    Features shape: {features.shape}")
            print(f"    Features range: [{features.min():.3f}, {features.max():.3f}]")
            
        except Exception as e:
            print(f"\n  ❌ Erreur observation {i+1}: {e}")
            logger.error(f"Erreur détaillée obs {i+1}: {e}", exc_info=True)
    
    print("\n✅ Test 1 terminé")
    return observations[0] if observations else None


def test_different_scenarios():
    """Test 2: Scénarios de jeu variés"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Scénarios variés (preflop, flop, turn, river)")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    
    scenarios = [
        # Scénario 1: Preflop premium
        {
            'name': 'Preflop AA BTN',
            'state': GameState(
                hole_cards=['As', 'Ah'],
                board=[],
                street='preflop',
                position='BTN',
                num_active_players=3,
                pot_size=150,
                stack=10000,
                big_blind=100,
                small_blind=50,
                amount_to_call=100,
                legal_actions=['fold', 'call', 'raise'],
                actions_this_street=[]
            )
        },
        
        # Scénario 2: Flop avec flush draw
        {
            'name': 'Flop flush draw',
            'state': GameState(
                hole_cards=['Kh', 'Qh'],
                board=['Jh', '9c', '4h'],
                street='flop',
                position='CO',
                num_active_players=2,
                pot_size=600,
                stack=9400,
                big_blind=100,
                small_blind=50,
                amount_to_call=400,
                legal_actions=['fold', 'call', 'raise'],
                actions_this_street=['bet_400']
            )
        },
        
        # Scénario 3: Turn avec paire
        {
            'name': 'Turn top pair',
            'state': GameState(
                hole_cards=['Ac', 'Kd'],
                board=['Ah', '9s', '4c', '2d'],
                street='turn',
                position='BB',
                num_active_players=2,
                pot_size=1200,
                stack=8000,
                big_blind=100,
                small_blind=50,
                amount_to_call=0,
                legal_actions=['check', 'bet'],
                actions_this_street=['check']
            )
        },
        
        # Scénario 4: River all-in
        {
            'name': 'River all-in situation',
            'state': GameState(
                hole_cards=['Qs', 'Qd'],
                board=['Kc', 'Jh', 'Tc', '9s', '2h'],
                street='river',
                position='SB',
                num_active_players=2,
                pot_size=3000,
                stack=1500,
                big_blind=100,
                small_blind=50,
                amount_to_call=1500,
                legal_actions=['fold', 'call'],
                actions_this_street=['bet_1500']
            )
        },
        
        # Scénario 5: Short stack preflop
        {
            'name': 'Short stack (10BB)',
            'state': GameState(
                hole_cards=['Td', 'Ts'],
                board=[],
                street='preflop',
                position='UTG',
                num_active_players=6,
                pot_size=150,
                stack=1000,
                big_blind=100,
                small_blind=50,
                amount_to_call=100,
                legal_actions=['fold', 'call', 'raise'],
                actions_this_street=[]
            )
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        state = scenario['state']
        
        try:
            features = extractor.extract(state)
            
            # Extraction de features clés
            feature_names = extractor.get_feature_names()
            
            key_indices = {
                'equity': feature_names.index('equity'),
                'spr_normalized': feature_names.index('spr_normalized'),
                'pot_odds': feature_names.index('pot_odds'),
                'stack_bb_normalized': feature_names.index('stack_bb_normalized'),
                'aggression_factor': feature_names.index('aggression_factor'),
                'position_normalized': feature_names.index('position_normalized'),
            }
            
            print(f"  Hole cards: {state.hole_cards}")
            print(f"  Board: {state.board if state.board else 'N/A'}")
            print(f"  Stack: {state.effective_stack_bb:.1f} BB")
            print(f"  Pot: {state.pot_size_bb:.1f} BB")
            print(f"  SPR: {state.spr:.2f}")
            print(f"  Features extraites:")
            
            for name, idx in key_indices.items():
                print(f"    {name:25s}: {features[idx]:.3f}")
            
            results.append({
                'scenario': scenario['name'],
                'features': features,
                'state': state
            })
            
            print("  ✅ OK")
            
        except Exception as e:
            print(f"  ❌ ERREUR: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def test_feature_consistency():
    """Test 3: Cohérence des features"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Cohérence et validité des features")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    
    # État de test
    state = GameState(
        hole_cards=['Kh', 'Kd'],
        board=['Ks', '9c', '4h'],
        street='flop',
        position='BTN',
        num_active_players=3,
        pot_size=500,
        stack=9500,
        big_blind=100,
        small_blind=50,
        amount_to_call=200,
        legal_actions=['fold', 'call', 'raise'],
        actions_this_street=['bet_200']
    )
    
    features = extractor.extract(state)
    
    # Tests de validité
    checks = []
    
    # 1. Aucun NaN
    has_nan = np.isnan(features).any()
    checks.append(('No NaN', not has_nan))
    print(f"  {'✅' if not has_nan else '❌'} No NaN: {not has_nan}")
    
    # 2. Valeurs dans [0, 1] pour les features normalisées
    normalized_features = [
        'equity', 'spr_normalized', 'pot_odds', 'stack_bb_normalized',
        'position_normalized', 'aggression_factor'
    ]
    
    for fname in normalized_features:
        if fname in feature_names:
            idx = feature_names.index(fname)
            val = features[idx]
            in_range = 0 <= val <= 1
            checks.append((f'{fname} in [0,1]', in_range))
            print(f"  {'✅' if in_range else '❌'} {fname:25s}: {val:.3f} {'✓' if in_range else '✗ OUT OF RANGE'}")
    
    # 3. Features one-hot somment à 1
    one_hot_groups = [
        (['street_preflop', 'street_flop', 'street_turn', 'street_river'], 'Street one-hot'),
        (['position_early', 'position_middle', 'position_late'], 'Position one-hot'),
        (['stack_short', 'stack_medium', 'stack_deep'], 'Stack category'),
    ]
    
    for features_list, group_name in one_hot_groups:
        indices = [feature_names.index(f) for f in features_list if f in feature_names]
        total = sum(features[idx] for idx in indices)
        is_valid = abs(total - 1.0) < 0.01
        checks.append((f'{group_name} sums to 1', is_valid))
        print(f"  {'✅' if is_valid else '❌'} {group_name:25s}: sum={total:.3f}")
    
    # 4. Cohérence SPR
    spr_calculated = state.spr
    spr_feature_idx = feature_names.index('spr_normalized')
    spr_feature = features[spr_feature_idx]
    print(f"\n  SPR cohérence:")
    print(f"    Calculé: {spr_calculated:.2f}")
    print(f"    Feature (normalized): {spr_feature:.3f}")
    print(f"    Feature (dénormalisé): {spr_feature * 20:.2f}")
    
    # 5. Cohérence pot odds
    pot_odds_calculated = state.pot_odds
    pot_odds_feature_idx = feature_names.index('pot_odds')
    pot_odds_feature = features[pot_odds_feature_idx]
    print(f"\n  Pot odds cohérence:")
    print(f"    Calculé: {pot_odds_calculated:.3f}")
    print(f"    Feature: {pot_odds_feature:.3f}")
    
    # Résumé
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n📊 Résumé: {passed}/{total} checks passés")
    
    return all(result for _, result in checks)

# tests/test_feature_extractor_full.py
# (Remplace juste la fonction create_sample_dataset)

def create_sample_dataset():
    """Test 4: Création d'un mini dataset pour XGBoost"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Création d'un dataset d'entraînement")
    print("=" * 70)
    
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 3})
    extractor = FeatureExtractor()
    
    # Définir les agents
    from rlcard.agents import RandomAgent
    agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
    env.set_agents(agents)
    
    dataset = {
        'features': [],
        'actions': [],
        'rewards': [],
        'game_states': []
    }
    
    num_games = 20
    
    print(f"\n🎮 Simulation de {num_games} parties...")
    
    for game_idx in range(num_games):
        trajectories, payoffs = env.run(is_training=False)
        
        # Debug première itération
        if game_idx == 0:
            print(f"\n🔍 Debug première partie:")
            print(f"  Nombre de joueurs: {len(trajectories)}")
            print(f"  Longueur trajectory player 0: {len(trajectories[0])}")
            
            # Afficher structure
            print(f"\n  Structure de la trajectory:")
            for i, elem in enumerate(trajectories[0][:6]):
                print(f"    [{i}] type: {type(elem).__name__:15s} - ", end="")
                if isinstance(elem, dict):
                    raw_obs = elem.get('raw_obs', {})
                    stage = raw_obs.get('stage', 'N/A')
                    hand = raw_obs.get('hand', [])
                    print(f"État ({stage}) - hand: {hand}")
                elif isinstance(elem, (int, np.integer)):
                    action_names = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise_pot', 4: 'all_in'}
                    print(f"Action = {elem} ({action_names.get(int(elem), 'unknown')})")
                else:
                    print(elem)
        
        # 🎯 Parcourir la trajectory: [state0, action0, state1, action1, ...]
        for player_id, trajectory in enumerate(trajectories):
            i = 0
            while i < len(trajectory):
                try:
                    # Lire l'état (doit être un dict)
                    if not isinstance(trajectory[i], dict):
                        i += 1
                        continue
                    
                    state_dict = trajectory[i]
                    
                    # Vérifier qu'on a bien obs et raw_obs
                    if 'obs' not in state_dict or 'raw_obs' not in state_dict:
                        i += 1
                        continue
                    
                    # Vérifier s'il y a une action qui suit
                    if i + 1 >= len(trajectory):
                        # Dernier état, pas d'action
                        break
                    
                    next_elem = trajectory[i + 1]
                    
                    # L'action doit être un int
                    if not isinstance(next_elem, (int, np.integer)):
                        i += 1
                        continue
                    
                    action = int(next_elem)
                    
                    # 🔧 FIX: Passer le dict complet, pas juste obs
                    game_state = RLCardAdapter.to_game_state(state_dict, env)
                    
                    # Extraction des features
                    features = extractor.extract(game_state)
                    
                    # Reward = payoff final du joueur
                    reward = payoffs[player_id]
                    
                    dataset['features'].append(features)
                    dataset['actions'].append(action)
                    dataset['rewards'].append(reward)
                    dataset['game_states'].append({
                        'street': game_state.street,
                        'position': game_state.position,
                        'stack_bb': game_state.effective_stack_bb
                    })
                    
                    # Avancer de 2 (état + action)
                    i += 2
                    
                except Exception as e:
                    if game_idx == 0:  # Debug seulement première game
                        logger.error(f"Erreur game {game_idx}, player {player_id}, index {i}: {e}")
                        import traceback
                        traceback.print_exc()
                    i += 1
                    continue
        
        if (game_idx + 1) % 5 == 0:
            print(f"  Progress: {game_idx + 1}/{num_games} parties ({len(dataset['features'])} samples collectés)")
    
    # Vérifier qu'on a collecté des données
    if len(dataset['features']) == 0:
        raise ValueError("Aucune donnée collectée - structure RLCard incompatible")
    
    # Conversion en arrays numpy
    X = np.array(dataset['features'])
    y = np.array(dataset['actions'])
    rewards = np.array(dataset['rewards'])
    
    print(f"\n✅ Dataset créé:")
    print(f"  Samples: {len(X)}")
    print(f"  Features shape: {X.shape}")
    print(f"  Actions: {len(np.unique(y))} types uniques")
    print(f"  Rewards: min={rewards.min():.2f}, mean={rewards.mean():.2f}, max={rewards.max():.2f}")
    
    # Distribution des actions
    action_names = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise_pot', 4: 'all_in'}
    print(f"\n📊 Distribution des actions:")
    for action_id in sorted(np.unique(y)):
        count = (y == action_id).sum()
        pct = 100 * count / len(y)
        action_name = action_names.get(int(action_id), f"action_{action_id}")
        print(f"  {action_name:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Distribution par street
    print(f"\n📊 Distribution par street:")
    streets = [gs['street'] for gs in dataset['game_states']]
    for street in ['preflop', 'flop', 'turn', 'river']:
        count = streets.count(street)
        pct = 100 * count / len(streets) if streets else 0
        print(f"  {street:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Stats sur les features
    print(f"\n📊 Stats sur les features:")
    print(f"  Min: {X.min():.3f}")
    print(f"  Max: {X.max():.3f}")
    print(f"  Mean: {X.mean():.3f}")
    print(f"  Std: {X.std():.3f}")
    print(f"  NaN: {np.isnan(X).sum()}")
    print(f"  Inf: {np.isinf(X).sum()}")
    
    # Sauvegarde
    print(f"\n💾 Sauvegarde du dataset...")
    os.makedirs('data', exist_ok=True)
    
    np.savez('data/sample_dataset.npz', 
             X=X, y=y, rewards=rewards,
             feature_names=extractor.get_feature_names())
    print("  ✅ Sauvegardé dans data/sample_dataset.npz")
    
    return X, y, rewards







def main():
    """Lance tous les tests"""
    print("\n" + "=" * 70)
    print("🚀 TESTS COMPLETS DU FEATURE EXTRACTOR")
    print("=" * 70)
    
    try:
        # Test 1
        obs = test_rlcard_integration()
        
        # Test 2
        scenarios = test_different_scenarios()
        
        # Test 3
        is_consistent = test_feature_consistency()
        
        # Test 4
        X, y, rewards = create_sample_dataset()
        
        print("\n" + "=" * 70)
        print("✅ TOUS LES TESTS TERMINÉS")
        print("=" * 70)
        
        print("\n📋 Résumé:")
        print(f"  ✅ Observations RLCard collectées: {len(obs)}")
        print(f"  ✅ Scénarios testés: {len(scenarios)}")
        print(f"  {'✅' if is_consistent else '❌'} Cohérence: {is_consistent}")
        print(f"  ✅ Dataset créé: {len(X)} samples")
        
        print("\n🎯 Prochaine étape: Entraîner un modèle XGBoost simple")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
