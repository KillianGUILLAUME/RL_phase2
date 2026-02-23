import os
import time
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.integration.sb3 import WandbCallback


from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



from training.sb3wrapper import PokerSB3Wrapper, PPOBotAgent
from training.callbacks import ProgressCallback, CheckpointCallback, SmartCheckpointCallback, PlottingCallback, save_sb3_with_version

from typing import Callable, List
from functools import partial

from dotenv import load_dotenv

load_dotenv()




def get_git_commit_hash():
    try:
        import subprocess
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
        return commit_hash.strip().decode('utf-8')
    except Exception:
        return "12ab34g"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Fonction nécessaire pour corriger le bug Python 3.10 -> 3.12
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- TA CONFIG ---
MY_CONFIG = {
    # 'total_timesteps': 40_000_000, #3 + 5 + 5M / training vs rule_2
    'total_timesteps': 27_000_000,
    'N_CPU': max(1, os.cpu_count() // 2) if os.cpu_count() else 8,
    'BATCH_SIZE': 512
}

# OPPONENT_TYPE = 'rule_v2'
# NUM_OPPONENTS = 5
OPPONNENTS =['rule', 'rule', 'rule_v2', 'rule_v2', 'output_bot.onnx']
NUM_OPPONENTS = len(OPPONNENTS)
WANDB_RUN_ID = "g7lnuxsr"


COMMIT_HASH = get_git_commit_hash()
TIMESTAMP = time.strftime("%Y%m%d-%H%M")
RUN_NAME = f"ppo-512x256x128-{OPPONNENTS}-{TIMESTAMP}"
SAVE_DIR = f"./models/rl_203_features/standard/resume_training/{RUN_NAME}"
LOG_DIR = f"./logs/rl_203_features/standard/resume_training/{RUN_NAME}"


TRAINING_CONFIG = {
    "algo": "MaskablePPO",
    "total_timesteps": MY_CONFIG['total_timesteps'],
    "opponents": OPPONNENTS,
    "num_opponents": NUM_OPPONENTS,
    "learning_rate": 5e-5,
    "n_steps": 2048,
    "batch_size": MY_CONFIG['BATCH_SIZE'],
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.05,
    'target_kl': 0.03,
    "features_dim": 203 
}




PPO_PARAMS = {
    "learning_rate": 5e-5,      
    "n_steps": 2048,            
    "batch_size": MY_CONFIG['BATCH_SIZE'],
    "n_epochs": 10,           
    "gamma": 0.95,            
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.05,
    'target_kl': 0.03,
    "verbose": 1,
}


def plot_training_results(log_dir, save_dir, num_timesteps):
    print(f"\n🎨 Génération du graphique timestep {num_timesteps}...")
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
        fig.suptitle(f'Self-Play Rotation — {num_timesteps} steps', fontsize=16)
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
        output_img = os.path.join(save_dir, f"training_summary_{num_timesteps}.png")
        plt.savefig(output_img)
    except Exception as e:
        print(f"⚠️ Erreur plot : {e}")

def make_env():
    """
    Fonction factory appelée par chaque processus CPU.
    Crée une instance indépendante de l'environnement.
    """
    try:
        env = PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, opponents_config=OPPONNENTS)
        return env
    except Exception as e:
        print(f"⚠️ Erreur création env ({OPPONNENTS}): {e}")
        print("➡️ Fallback sur 'random' pour éviter le crash.")
        return PokerSB3Wrapper(num_opponents=NUM_OPPONENTS, opponents_config='random')


if __name__ == '__main__':
    print(f"start training")
    print(f"   - Cœurs CPU : {MY_CONFIG['N_CPU']}")
    print(f"   - Adversaires : {OPPONNENTS}")
    print(f"   - Total Steps : {MY_CONFIG['total_timesteps']}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    vec_env = make_vec_env(
        make_env, 
        n_envs=MY_CONFIG['N_CPU'], 
        vec_env_cls=SubprocVecEnv
    )
    
    pretrained_path= 'models/rl_203_features/standard/resume_training/ppo_sb3_poker_23510624_steps.zip'
    use_pretrained_model = True
    
    if use_pretrained_model:
        print(f"♻️ CHARGEMENT DU MODÈLE PRÉ-ENTRAÎNÉ : {pretrained_path}")
        
        def constant_schedule(value: float) -> Callable[[float], float]:
            def func(progress_remaining: float) -> float:
                return value
            return func

        custom_objects = {
            "learning_rate": constant_schedule(5e-5),
            "lr_schedule": constant_schedule(5e-5),
            "clip_range": constant_schedule(0.15),
            "ent_coef": 0.05,
            "target_kl": 0.03,
            "n_epochs": 10,
            "n_steps": 2048,
            "batch_size": MY_CONFIG['BATCH_SIZE'],
            "gamma": 0.95,
            "gae_lambda": 0.95
        }

        # On charge le modèle en remplaçant les parties cassées
        model = MaskablePPO.load(
            pretrained_path, 
            env=vec_env, 
            tensorboard_log=LOG_DIR,
            custom_objects=custom_objects,
            print_system_info=True
        )
        
        model.learning_rate = constant_schedule(5e-5)
        model.clip_range = constant_schedule(0.15)
        model.ent_coef = 0.05
        model.target_kl = 0.03
        model.n_epochs = 10
        model.gamma = 0.95
        model.gae_lambda = 0.95
        model.n_steps = 2048
        model.batch_size = MY_CONFIG['BATCH_SIZE']

    else:
        print("Création d'un modèle from scratch")
        policy_kwargs = dict(
            net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
        )

        tensorboard_log_dir = f'{LOG_DIR}/tensorboard'
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir,
            **PPO_PARAMS
        )
    
    smart_callback = SmartCheckpointCallback(
        save_freq=max(500_000 // MY_CONFIG['N_CPU'], 1), 
        save_dir=SAVE_DIR, 
        config=TRAINING_CONFIG
    )

    plotting_callback = PlottingCallback(
        log_dir=LOG_DIR,
        save_dir=SAVE_DIR,
        plot_func=plot_training_results,
        plot_freq=500_000,
        verbose=0
    )

    callbacks_list = [smart_callback, plotting_callback]

    if os.getenv("WANDB_API_KEY"):
        print(f"🌊 Initialisation Weights & Biases...")
        
        init_kwargs = {
            "project": "poker",
            "entity": "killian-guillaume-personal",
            "config": TRAINING_CONFIG,
            "sync_tensorboard": True,
            "monitor_gym": True,
            "save_code": True,
        }
        
        if WANDB_RUN_ID:
            print(f"➡️ Reprise du run W&B : {WANDB_RUN_ID}")
            init_kwargs["id"] = WANDB_RUN_ID
            init_kwargs["resume"] = "must"
        else:
            init_kwargs["name"] = RUN_NAME

        run = wandb.init(**init_kwargs)
    else:
        print("⚠️ W&B désactivé (Pas de clé trouvée).")

    callbacks_list = [smart_callback, plotting_callback]

    wandb_callback = WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=None,
                    verbose=0,
                )

    
    callbacks_list.append(wandb_callback)
        
    
    try:
        do_reset = not use_pretrained_model
        
        model.learn(
            total_timesteps=MY_CONFIG['total_timesteps'], 
            callback=callbacks_list,
            progress_bar=True,
            reset_num_timesteps=do_reset
        )
        
        save_sb3_with_version(model, "final_model_rule", SAVE_DIR, "end", TRAINING_CONFIG)

        final_std_path = os.path.join(SAVE_DIR, "final_model_rule.zip")
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
        wandb.finish()
        # Génération du graphique
        # On attend un peu que Tensorboard ait fini d'écrire sur le disque
        time.sleep(15) 
        plot_training_results(LOG_DIR, SAVE_DIR, MY_CONFIG['total_timesteps'])
        
        print(f"\nplot saved in: {SAVE_DIR}")