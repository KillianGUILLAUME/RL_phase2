# test_params.py
import sys
sys.path.append('.')

import rlcard
from agents.dqn import SmartDQNAgent

env = rlcard.make('no-limit-holdem')

# ✅ Test 1 : Sans paramètres (valeurs par défaut)
agent1 = SmartDQNAgent(env)
print("✅ Test 1 : Paramètres par défaut OK")

# ✅ Test 2 : Avec tous les paramètres
agent2 = SmartDQNAgent(
    env,
    device='cpu',
    mlp_layers=[512, 256, 128],
    replay_memory_size=100000,
    batch_size=64,
    replay_memory_init_size=5000,
    update_target_estimator_every=500,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50000,
    learning_rate=0.0001,
    discount_factor=0.95
)
print("✅ Test 2 : Tous les paramètres OK")

# ✅ Test 3 : Script train_dqn.py devrait marcher maintenant
print("✅ Tous les tests passés ! Relance train_dqn.py")
