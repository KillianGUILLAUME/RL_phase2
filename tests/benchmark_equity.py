import time
import random
from features.card_features import UtilsCardFeatures

def benchmark():
    print("==================================================")
    print("🎰 POKER EQUITY BENCHMARK")
    print("==================================================\n")
    
    # Mock des sets pour éviter les erreurs Preflop
    premium = {'AA', 'KK', 'QQ', 'JJ', 'AKs'}
    strong = {'TT', '99', '88', 'AQs', 'AJs', 'KQs', 'AKo', 'AQo'}
    ucf = UtilsCardFeatures(premium_hands=premium, strong_hands=strong)
    
    ranks = '23456789TJQKA'
    suits = 'shdc'
    full_deck = [r+s for r in ranks for s in suits]
    
    scenarios = []
    print("🎲 Génération de 500 scénarios aléatoires (Preflop, Flop, Turn, River)...")
    for _ in range(500):
        deck = full_deck.copy()
        random.shuffle(deck)
        
        hole_cards = [deck.pop(), deck.pop()]
        board_len = random.choice([0, 3, 4, 5])
        board = [deck.pop() for _ in range(board_len)]
        num_opponents = random.randint(1, 5)
        
        scenarios.append((hole_cards, board, num_opponents))

    # 1. Benchmark Heuristic Equity (First Pass = Uncached)
    print("\n🚀 --- 1. Équité Heuristique (_estimate_equity) ---")
    start = time.time()
    for hc, b, opps in scenarios:
        ucf._estimate_equity(hc, b, opps)
    end = time.time()
    heur_time = end - start
    print(f"⏳ Temps pour 500 appels uniques : {heur_time:.4f} secondes")
    
    # 2. Benchmark Heuristic Equity (Second Pass = Cached)
    start = time.time()
    for hc, b, opps in scenarios:
        ucf._estimate_equity(hc, b, opps)
    end = time.time()
    heur_time_cached = end - start
    print(f"⚡ Temps pour 500 appels en CACHE (LRU) : {heur_time_cached:.4f} secondes")

    # 3. Benchmark Monte Carlo Equity (100 sims)
    print("\n🎲 --- 2. Monte Carlo Fast (100 simulations) ---")
    start = time.time()
    for hc, b, opps in scenarios:
        ucf._monte_carlo_equity(hc, b, opps, num_simulations=100)
    end = time.time()
    mc_100_time = end - start
    print(f"⏳ Temps pour 500 appels (50,000 runs totaux) : {mc_100_time:.4f} secondes")
    
    # 4. Compare outputs for a few scenarios
    print("\n📊 --- Comparaison de précision sur 5 scénarios concrets ---")
    for i in range(5):
        hc, b, opps = scenarios[i]
        heur_val = ucf._estimate_equity(hc, b, opps)
        mc_val   = ucf._monte_carlo_equity(hc, b, opps, num_simulations=1000) # Haute précision pour la démo
        
        b_str = str(b) if b else "[] (Preflop)"
        print(f"Joueurs=1+{opps} | Mains: {hc} | Board: {b_str}")
        print(f"   => Heuristique : {heur_val*100:.1f}%")
        print(f"   => Monte Carlo : {mc_val*100:.1f}% (Diff: {abs(heur_val-mc_val)*100:.1f}%)")
        print("-" * 50)

if __name__ == "__main__":
    benchmark()
