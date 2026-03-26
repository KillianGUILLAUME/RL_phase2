import os
import glob
import numpy as np
import scipy.optimize
import rlcard
import time
from concurrent.futures import ProcessPoolExecutor
from training.sb3wrapper import PPOBotAgent

ARCHIVE_DIR = "models/psro_archive"
NUM_EVAL_GAMES = 30000

def compute_nash_equilibrium(payoff_matrix):
    """
    Computes the Nash equilibrium of a zero-sum game using linear programming.
    payoff_matrix[i, j] is the payoff to P1 when P1 plays strategy i and P2 plays strategy j.
    """
    num_strats = payoff_matrix.shape[0]
    if num_strats == 1:
        return np.array([1.0])
        
    # Minimize V subject to:
    # V - sum_i p_i * payoff_matrix[i, j] >= 0 for all j
    # sum p = 1, p >= 0
    #
    # Translated to Scipy linprog standard form (minimize c^T x subject to A_ub x <= b_ub and A_eq x = b_eq):
    # Let x = [V, p_1, ..., p_n]
    # Minimize x[0] -> c = [1, 0, 0, ..., 0]
    # Constraint: -V + sum_i p_i * M[i, j] <= 0 -> A_ub[:, 0] = -1, A_ub[:, 1:] = M^T
    
    c = np.zeros(num_strats + 1)
    c[0] = -1.0 # Maximize V -> Minimize -V
    
    A_ub = np.zeros((num_strats, num_strats + 1))
    A_ub[:, 0] = 1.0
    A_ub[:, 1:] = -payoff_matrix.T
    
    b_ub = np.zeros(num_strats)
    
    A_eq = np.zeros((1, num_strats + 1))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(None, None)] + [(0, 1)] * num_strats
    
    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        probs = res.x[1:]
        # zero out extremely small probabilities and re-normalize
        probs[probs < 1e-4] = 0
        probs /= probs.sum()
        return probs
    else:
        print("⚠️ Failed to find Nash Equilibrium, falling back to Uniform.")
        return np.ones(num_strats) / num_strats


def play_match(model_path_1, model_path_2, num_games=NUM_EVAL_GAMES):
    """
    Evaluates two models against each other in a heads-up rotating tournament.
    Returns the average BB/hand payoff for model 1.
    """
    env = rlcard.make('no-limit-holdem', config={'game_num_players': 2})
    
    agent1 = PPOBotAgent(model_path_1, env)
    agent2 = PPOBotAgent(model_path_2, env)
    
    agents = [agent1, agent2]
    
    payoffs_p1 = 0.0
    
    for game_idx in range(num_games):
        shift = game_idx % 2
        rotated_agents = agents[-shift:] + agents[:-shift]
        env.set_agents(rotated_agents)
        
        trajectories, payoffs = env.run(is_training=False)
        
        # Determine payoff for agent1 based on rotation
        if shift == 0:
            payoffs_p1 += payoffs[0]
        else:
            payoffs_p1 += payoffs[1]
            
    # Convert from chips to Big Blinds
    env_big_blind = 2
    bb_per_hand = (payoffs_p1 / num_games) / env_big_blind
    return bb_per_hand


def evaluate_and_solve():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # 1. Discover models
    model_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.zip")))
    num_models = len(model_files)
    
    if num_models == 0:
        print("Aucun modèle dans l'archive PSRO.")
        return
        
    print(f"📊 PSRO Evaluator: {num_models} modèles détectés.")
    
    # 2. Load or initialize Payoff Matrix
    matrix_path = os.path.join(ARCHIVE_DIR, "payoff_matrix.npy")
    if os.path.exists(matrix_path):
        old_matrix = np.load(matrix_path)
        old_size = old_matrix.shape[0]
    else:
        old_matrix = np.empty((0, 0))
        old_size = 0
        
    if old_size > num_models:
        print("Anomalie : matrice plus grande que le nombre de modèles. Réinitialisation.")
        old_matrix = np.empty((0, 0))
        old_size = 0
        
    matrix = np.zeros((num_models, num_models))
    if old_size > 0:
        matrix[:old_size, :old_size] = old_matrix
        
    # 3. Evaluate new match-ups (we only need to evaluate upper triangle since zero-sum: M[i, j] = -M[j, i])
    matches_to_play = []
    for i in range(num_models):
        for j in range(i + 1, num_models):
            if i >= old_size or j >= old_size:
                matches_to_play.append((i, j))
                
    if len(matches_to_play) > 0:
        print(f"⚔️ {len(matches_to_play)} nouveaux affrontements Heads-up à évaluer ({NUM_EVAL_GAMES} mains chacun)...")
        # Optimization: we compute matchups sequentially to avoid pickling the environments 
        # (rlcard envs don't need heavy MP if num_games is only 2000, it takes a few seconds)
        for idx_pair, (i, j) in enumerate(matches_to_play):
            print(f"   [{idx_pair+1}/{len(matches_to_play)}] Evaluaton {os.path.basename(model_files[i])} VS {os.path.basename(model_files[j])}...")
            
            payoff_1_vs_2 = play_match(model_files[i], model_files[j])
            matrix[i, j] = payoff_1_vs_2
            matrix[j, i] = -payoff_1_vs_2 # Zero-sum asymmetry
            
        # Save updated matrix
        np.save(matrix_path, matrix)
        print("✅ Matrice des gains mise à jour et sauvegardée.")
        
    print(f"\n🧩 Matrice Empirique des gains (Ligne VS Colonne, en BB/main):")
    # Affichage propre via Numpy
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(matrix)
    print("------------------------------------------------------------")
        
    # 4. Compute Nash Equilibrium
    print("\n🧮 Calcul de l'Équilibre de Nash sur la Matrice des gains...")
    nash_probs = compute_nash_equilibrium(matrix)
    
    print("📈 Distribution Idéale de la Méta-Stratégie (Nash):")
    for i, p in enumerate(nash_probs):
        if p > 0.001:
            print(f"   - {os.path.basename(model_files[i])}: {p*100:.1f}%")
            
    # Save weights
    weights_path = os.path.join(ARCHIVE_DIR, "nash_weights.npy")
    np.save(weights_path, nash_probs)
    print(f"✅ Poids Nash sauvegardés ({weights_path}).")
    

if __name__ == "__main__":
    evaluate_and_solve()
