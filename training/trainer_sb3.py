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


from training.sb3wrapper import PokerSB3Wrapper
from training.callbacks import ProgressCallback, CheckpointCallback, SmartCheckpointCallback



from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Fonction nécessaire pour corriger le bug Python 3.10 -> 3.12
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- TA CONFIG ---
MY_CONFIG = {
    'total_timesteps': 3_000_000,
    'N_CPU': os.cpu_count(),
    'BATCH_SIZE': 512
}

OPPONENT_TYPE = 'xgb' 
NUM_OPPONENTS = 5


RUN_NAME = f"PPO_Poker_{OPPONENT_TYPE}_{int(time.time())}"
SAVE_DIR = f"./models/{RUN_NAME}"
LOG_DIR = f"./logs/{RUN_NAME}"

TRAINING_CONFIG = {
    "algo": "MaskablePPO",
    "total_timesteps": MY_CONFIG['total_timesteps'],
    "opponents": OPPONENT_TYPE,
    "num_opponents": NUM_OPPONENTS,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": MY_CONFIG['BATCH_SIZE'],
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "features_dim": 87  # Taille de ton vecteur de features
}




PPO_PARAMS = {
    "learning_rate": 3e-4,      # Vitesse d'apprentissage standard
    "n_steps": 1024,            # Nombre de steps par CPU avant update (plus long = plus stable)
    "batch_size": MY_CONFIG['BATCH_SIZE'],          # Taille du batch pour l'optimisation GPU/CPU
    "n_epochs": 10,             # Combien de fois on révise les données
    "gamma": 0.995,             # Importance du futur (le poker se joue sur le long terme)
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,           # ✨ CRUCIAL : Force l'IA à explorer (bluffer/varier)
    "verbose": 1,
}

from training.callbacks import save_sb3_with_version

def plot_training_results(log_dir, save_dir):
    print(f"\n🎨 Génération du graphique final...")
    
    # Trouver le fichier de logs le plus récent
    search_path = os.path.join(log_dir, "**", "events.out.tfevents*")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("❌ Aucun log Tensorboard trouvé pour le graphique.")
        return

    latest_file = max(files, key=os.path.getmtime)
    print(f"   Lecture des logs : {os.path.basename(latest_file)}")
    
    try:
        event_acc = EventAccumulator(latest_file)
        event_acc.Reload()
        
        # Tags à tracer
        tags_to_plot = {
            'rollout/ep_rew_mean': 'Récompense Moyenne (Reward)',
            'train/entropy_loss': 'Entropie (Exploration)',
            'train/value_loss': 'Value Loss (Critique)',
            'rollout/ep_len_mean': 'Durée moyenne (Tours)'
        }
        
        available_tags = event_acc.Tags()['scalars']
        
        # Création de la figure (2x2 plots)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Entraînement Poker Bot - {OPPONENT_TYPE}', fontsize=16)
        axes = axes.flatten()
        
        for i, (tag, title) in enumerate(tags_to_plot.items()):
            if tag in available_tags:
                events = event_acc.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                axes[i].plot(steps, values, linewidth=2)
                axes[i].set_title(title)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlabel("Steps")
            else:
                axes[i].text(0.5, 0.5, "Données manquantes", ha='center')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarde de l'image
        output_img = os.path.join(save_dir, "training_summary.png")
        plt.savefig(output_img)
        print(f"✅ Graphique sauvegardé : {output_img}")
        
    except Exception as e:
        print(f"⚠️ Erreur lors du plotting : {e}")


def make_env():
    """
    Fonction factory appelée par chaque processus CPU.
    Crée une instance indépendante de l'environnement.
    """
    # On gère le cas où le modèle XGB n'existe pas encore
    try:
        env = PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, str_opponents=OPPONENT_TYPE)
        return env
    except Exception as e:
        print(f"⚠️ Erreur création env ({OPPONENT_TYPE}): {e}")
        print("➡️ Fallback sur 'random' pour éviter le crash.")
        return PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, str_opponents='random')


if __name__ == '__main__':
    print(f"start training")
    print(f"   - Cœurs CPU : {MY_CONFIG['N_CPU']}")
    print(f"   - Adversaires : {NUM_OPPONENTS}x {OPPONENT_TYPE}")
    print(f"   - Total Steps : {MY_CONFIG['total_timesteps']}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(
        make_env, 
        n_envs=MY_CONFIG['N_CPU'], 
        vec_env_cls=SubprocVecEnv
    )
    
    # 2. Initialisation du modèle MaskablePPO
    # On utilise MlpPolicy car tes features (87 floats) sont un vecteur plat
    # --- MODIFICATION ICI : CHARGEMENT DU MODÈLE ---
    pretrained_path = "models/rl/PPO_Poker_random_1770672362/FINAL_MODEL_end_steps.zip"
    
    if os.path.exists(pretrained_path):
        print(f"♻️ CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ : {pretrained_path}")
        
        # --- CORRECTION DU BUG PYTHON 3.10 vs 3.12 ---
        # On force la réécriture des fonctions qui font planter le chargement
        custom_objects = {
            "learning_rate": 3e-4,
            "lr_schedule": linear_schedule(3e-4),
            "clip_range": linear_schedule(0.2)
        }
        
        # On charge le modèle en remplaçant les parties cassées
        model = MaskablePPO.load(
            pretrained_path, 
            env=vec_env, 
            tensorboard_log=LOG_DIR,
            custom_objects=custom_objects, # <-- C'est ça qui répare tout !
            print_system_info=True
        )
        
        # On s'assure que les hyperparamètres sont bien mis à jour
        model.learning_rate = linear_schedule(3e-4)
        model.clip_range = linear_schedule(0.2)
        model.ent_coef = 0.01
    else:
        print("👶 Aucun modèle trouvé, création d'un nouveau modèle (FROM SCRATCH)...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            tensorboard_log=LOG_DIR,
            **PPO_PARAMS
        )
    
    smart_callback = SmartCheckpointCallback(
        save_freq=max(5000 // MY_CONFIG['N_CPU'], 1), 
        save_dir=SAVE_DIR, 
        config=TRAINING_CONFIG
    )

    callbacks = [smart_callback]

    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=MY_CONFIG['total_timesteps'], 
            callback=callbacks,
            progress_bar=True
        )
        
        # 5. Fin & Sauvegarde Finale
        duration = time.time() - start_time
        print(f"\n✅ Entraînement terminé en {duration/60:.1f} minutes.")
        
        save_sb3_with_version(model, "FINAL_MODEL", SAVE_DIR, "end", TRAINING_CONFIG)

        final_std_path = os.path.join(SAVE_DIR, "final_model_standard.zip")
        model.save(final_std_path)
        print(f"✅ Modèle standard sauvegardé : {final_std_path}")
        
    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu manuellement.")
        save_sb3_with_version(model, "INTERRUPTED_MODEL", SAVE_DIR, "interrupted", TRAINING_CONFIG)
        
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        raise e
    finally:
        vec_env.close()
        
        # Génération du graphique
        # On attend un peu que Tensorboard ait fini d'écrire sur le disque
        time.sleep(15) 
        plot_training_results(LOG_DIR, SAVE_DIR)
        
        print(f"\nplot saved in: {SAVE_DIR}")