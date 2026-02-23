import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_state import GameState
from features.feature_builder import FeatureExtractor_v2

def check_scenario(name, state):
    print(f"\n{'='*50}")
    print(f"Scénario: {name}")
    print(f"{'='*50}")
    print(f"Hole cards: {state.hole_cards}")
    print(f"Board: {state.board}")
    print(f"Street: {state.street}, Pos: {state.position}")
    print(f"Pot: {state.pot_size}, Stack: {state.stack}, To call: {state.amount_to_call}")
    
    extractor = FeatureExtractor_v2()
    features = extractor.extract(state)
    names = extractor.get_feature_names()
    
    print("\n--- Card Features (0-21) ---")
    for i in range(22):
        print(f"{names[i]:30s} : {features[i]:.4f}")
        
    print("\n--- Game Theory Features (67-98) ---")
    for i in range(67, 99):
        if i < len(names):
            print(f"{names[i]:30s} : {features[i]:.4f}")

if __name__ == '__main__':
    # 1. Preflop Premium
    check_scenario("Preflop AA BTN", GameState(
        hole_cards=['As', 'Ah'], board=[], street='preflop', position='BTN',
        num_active_players=3, pot_size=150, stack=10000, big_blind=100, small_blind=50,
        amount_to_call=100, legal_actions=['fold', 'call', 'raise'], actions_this_street=[],
        all_actions_history=[]
    ))
    
    # 2. Flop Flush Draw + Overcards vs un bet
    check_scenario("Flop AhKh sur Jc 9c 4h face à un bet", GameState(
        hole_cards=['Ah', 'Kh'], board=['Jc', '9c', '4h'], street='flop', position='CO',
        num_active_players=2, pot_size=600, stack=9400, big_blind=100, small_blind=50,
        amount_to_call=400, legal_actions=['fold', 'call', 'raise'], actions_this_street=['bet_400'],
        all_actions_history=[]
    ))
    
    # 3. Turn Nut Flush face à check
    check_scenario("Turn Nut Flush (Ah Th sur 9h 4h 2h)", GameState(
        hole_cards=['Ah', 'Th'], board=['9h', '4h', '2h', 'Kc'], street='turn', position='BTN',
        num_active_players=2, pot_size=1500, stack=8500, big_blind=100, small_blind=50,
        amount_to_call=0, legal_actions=['check', 'raise'], actions_this_street=['check'],
        all_actions_history=[]
    ))
