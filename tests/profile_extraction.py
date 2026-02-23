import time
import numpy as np
from core.game_state import GameState
from features.feature_builder import FeatureExtractor_v2

def run_profiling(iterations=5000):
    state = GameState(
        hole_cards=['As', 'Kh'], board=['Jc', '9c', '4h'], street='flop', position='CO',
        num_active_players=2, pot_size=600, stack=9400, big_blind=100, small_blind=50,
        amount_to_call=400, legal_actions=['fold', 'call', 'raise'], actions_this_street=['bet_400'],
        all_actions_history=['raise_200', 'call_200', 'flop_starts', 'bet_400']
    )
    
    extractor = FeatureExtractor_v2()
    
    # Warmup cache
    for _ in range(10):
        extractor.extract(state)
        
    t0 = time.time()
    for _ in range(iterations):
        extractor.extract(state)
    t1 = time.time()
    
    total_time = t1 - t0
    ms_per_iter = (total_time / iterations) * 1000
    
    print(f"Total time for {iterations} extractions: {total_time:.4f}s")
    print(f"Time per extraction: {ms_per_iter:.4f} ms")
    
    # Profiler for bottlenecks if it's > 0.5ms
    if ms_per_iter > 0.5:
        import cProfile
        print("\n--- Running cProfile ---")
        cProfile.runctx('for _ in range(1000): extractor.extract(state)', globals(), locals(), sort='cumtime')

if __name__ == '__main__':
    run_profiling()
