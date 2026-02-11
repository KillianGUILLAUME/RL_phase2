import os
import time
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb
from wandb.integration.sb3 import WandbCallback

from training.sb3wrapper import PokerSB3Wrapper, PPOBotAgent
from training.callbacks import ProgressCallback, CheckpointCallback, SmartCheckpointCallback, save_sb3_with_version

from typing import Callable

def get_git_commit_hash():
    try:
        # On demande à git le hash court du dernier commit (HEAD)
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return commit_hash.strip().decode('utf-8')
    except Exception:
        return "no_git"

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """ Fonction nécessaire pour corriger le bug Python 3.10 -> 3.12 """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- TA CONFIG ---
MY_CONFIG = {
    'total_timesteps': 2_000, # <--- PHASE 3 : On augmente un peu la durée
    'N_CPU': os.cpu_count(),
    'BATCH_SIZE': 512 # Tu peux tenter 1024 ou 2048 si tu as beaucoup de RAM, sinon 512 c'est bien
}

OPPONENT_TYPE = 'models/rl/PPO_Poker_xgb_1770802762/FINAL_MODEL_end_steps.zip' 
NUM_OPPONENTS = 5

COMMIT_HASH = get_git_commit_hash()
TIMESTAMP = time.strftime("%Y%m%d-%H%M")

RUN_NAME = f"PPO_Poker_SELF_PLAY_{TIMESTAMP}_{COMMIT_HASH}"
SAVE_DIR = f"./models/{RUN_NAME}"
LOG_DIR = f"./logs/{RUN_NAME}"

# <--- PHASE 3 : Hyperparamètres affinés (Fine-Tuning)
TRAINING_CONFIG = {
    "algo": "MaskablePPO",
    "total_timesteps": MY_CONFIG['total_timesteps'],
    "opponents": OPPONENT_TYPE,
    "num_opponents": NUM_OPPONENTS,
    "learning_rate": 2.0e-4, # <--- On ralentit l'apprentissage (c'était 3e-4)
    "n_steps": 2048,         # <--- On augmente la vision (plus stable)
    "batch_size": MY_CONFIG['BATCH_SIZE'],
    "n_epochs": 10,
    "gamma": 0.999,          # <--- On vise le très long terme (0.995 -> 0.999)
    "gae_lambda": 0.95,
    "clip_range": 0.15,      # <--- On réduit la zone de modif (0.2 -> 0.15) pour ne pas casser ce qui marche
    "ent_coef": 0.005,       # <--- Moins d'aléatoire (0.01 -> 0.005), on consolide
    "features_dim": 87
}

PPO_PARAMS = {
    "learning_rate": 2.0e-4, # <--- PHASE 3
    "n_steps": 2048,         # <--- PHASE 3
    "batch_size": MY_CONFIG['BATCH_SIZE'],
    "n_epochs": 10,
    "gamma": 0.999,          # <--- PHASE 3
    "gae_lambda": 0.95,
    "clip_range": 0.15,      # <--- PHASE 3
    "ent_coef": 0.005,       # <--- PHASE 3
    "verbose": 1,
}



def plot_training_results(log_dir, save_dir):
    print(f"\n🎨 Génération du graphique final...")
    search_path = os.path.join(log_dir, "**", "events.out.tfevents*")
    files = glob.glob(search_path, recursive=True)
    if not files: return

    latest_file = max(files, key=os.path.getmtime)
    try:
        event_acc = EventAccumulator(latest_file)
        event_acc.Reload()
        tags_to_plot = {
            'rollout/ep_rew_mean': 'Récompense Moyenne',
            'train/entropy_loss': 'Entropie',
            'train/value_loss': 'Value Loss',
            'rollout/ep_len_mean': 'Durée moyenne'
        }
        available_tags = event_acc.Tags()['scalars']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Phase 3 : Fine-Tuning - {OPPONENT_TYPE}', fontsize=16)
        axes = axes.flatten()
        for i, (tag, title) in enumerate(tags_to_plot.items()):
            if tag in available_tags:
                events = event_acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                axes[i].plot(steps, values, linewidth=2)
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_img = os.path.join(save_dir, "training_summary.png")
        plt.savefig(output_img)
    except Exception as e:
        print(f"⚠️ Erreur plot : {e}")

def make_env():
    try:
        env = PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, str_opponents=OPPONENT_TYPE)
        return env
    except Exception as e:
        return PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, str_opponents='random')

if __name__ == '__main__':
    print(f"🔥 START PHASE 3 : FINE-TUNING")
    print(f"   - Cœurs CPU : {MY_CONFIG['N_CPU']}")
    print(f"   - Steps : {MY_CONFIG['total_timesteps']}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(make_env, n_envs=MY_CONFIG['N_CPU'], vec_env_cls=SubprocVecEnv)
    
    pretrained_path = "models/rl/PPO_Poker_xgb_1770802762/FINAL_MODEL_end_steps.zip"

    if os.getenv("WANDB_KEY"):
        print(f"🌊 Initialisation Weights & Biases...")
        run = wandb.init(
            project="Poker-Bot-RL",  # Nom de ton projet sur le site
            name=RUN_NAME,           # Nom de cette expérience précise
            config=TRAINING_CONFIG,  # Il enregistre tous tes hyperparamètres !
            sync_tensorboard=True,   # Important pour capturer les métriques SB3
            monitor_gym=True,        # Capture les vidéos si dispo (pas ici, mais bon)
            save_code=True,          # Sauvegarde ton code pour la postérité
        )
    else:
        print("⚠️ W&B désactivé (Pas de clé trouvée).")
        run = None
    
    if os.path.exists(pretrained_path):
        print(f"♻️ CHARGEMENT DU CHAMPION : {pretrained_path}")
        
        custom_objects = {
            "learning_rate": 2.0e-4,
            "lr_schedule": linear_schedule(2.0e-4),
            "clip_range": linear_schedule(0.15)
        }
        
        model = MaskablePPO.load(
            pretrained_path, 
            env=vec_env, 
            tensorboard_log=LOG_DIR,
            custom_objects=custom_objects,
            print_system_info=True
        )
        
        model.learning_rate = linear_schedule(2.0e-4)
        model.clip_range = linear_schedule(0.15)
        model.ent_coef = 0.005
        model.n_epochs = 10
        model.gamma = 0.999
        
    else:
        print(f"❌ ERREUR CRITIQUE : Je ne trouve pas le modèle champion : {pretrained_path}")
        print("Arrêt immédiat pour ne pas écraser le travail.")
        exit()
    
    # On sauvegarde souvent (toutes les ~5 min)
    smart_callback = SmartCheckpointCallback(
        save_freq=max(50000 // MY_CONFIG['N_CPU'], 1), 
        save_dir=SAVE_DIR, 
        config=TRAINING_CONFIG
    )

    callbacks_list = [smart_callback]

    if run is not None:
        wandb_callback = WandbCallback(
            gradient_save_freq=1000, # Sauvegarde les gradients (pour voir si ça explose)
            model_save_path=f"models/{RUN_NAME}", # Sauvegarde le modèle dans le cloud W&B
            verbose=2,
        )
        callbacks_list.append(wandb_callback)

    # Affichage calme
    try:
        model.learn(
            total_timesteps=MY_CONFIG['total_timesteps'], 
            callback=callbacks_list,
            log_interval=100,
            progress_bar=True
        )
        
        save_sb3_with_version(model, "FINAL_MODEL_SELF_PLAY", SAVE_DIR, "end", TRAINING_CONFIG)
        model.save(os.path.join(SAVE_DIR, "final_model_self_play.zip"))
        print(f"✅ Phase 3 Terminée.")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrompu.")
        save_sb3_with_version(model, "INTERRUPTED_SELF_PLAY", SAVE_DIR, "interrupted", TRAINING_CONFIG)
    finally:
        vec_env.close()
        if run is not None:
            wandb.finish()
        time.sleep(5) 
        plot_training_results(LOG_DIR, SAVE_DIR)