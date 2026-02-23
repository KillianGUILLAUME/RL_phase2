import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3.common.logger import configure
import wandb
from wandb.integration.sb3 import WandbCallback

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
    """ Fonction nécessaire pour corriger le bug Python 3.10 -> 3.12 """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ══════════════════════════════════════════════════════════════════════
#                      CONFIGURATION SELF-PLAY
# ══════════════════════════════════════════════════════════════════════

# --- Training global ---
PHASE_STEPS = 3_000_000            # Steps par phase
NUM_PHASES = 5                     # 5 + 1 phases = 6M total
# En RL, SubprocVecEnv est CPU-bound. Les vCPUs AWS sont hyperthreadés (2 vCPU = 1 coeur physique).
# Il faut utiliser les coeurs physiques pour éviter les contentions de cache.
N_CPU = max(1, os.cpu_count() // 2) if os.cpu_count() else 8
BATCH_SIZE = 512

FIXED_OPPONENTS = ['rule', 'rule_v2']
INITIAL_PPO_OPPONENT = 'models/rl_203_features/standard/resume_training/ppo_sb3_poker_27510624_steps.zip'
PRETRAINED_HERO = 'models/rl_203_features/standard/resume_training/ppo_sb3_poker_27510624_steps.zip'
SEED_MODEL = 'models/rl_203_features/standard/resume_training/ppo_sb3_poker_27510624_steps.zip'

NUM_PPO_SLOTS = 3

# --- Naming ---
COMMIT_HASH = get_git_commit_hash()
TIMESTAMP = time.strftime("%Y%m%d-%H%M")
RUN_NAME = f"selfplay-rotation-{TIMESTAMP}"
SAVE_DIR = f"./models/rl_99_features/self_play_rotation/{RUN_NAME}"
LOG_DIR = f"./logs/rl_99_features/self_play_rotation/{RUN_NAME}"

TRAINING_CONFIG = {
    "algo": "MaskablePPO",
    "strategy": "self-play-rotation",
    "total_timesteps": PHASE_STEPS * NUM_PHASES,
    "phase_steps": PHASE_STEPS,
    "num_phases": NUM_PHASES,
    "fixed_opponents": FIXED_OPPONENTS,
    "initial_ppo_opponent": INITIAL_PPO_OPPONENT,
    "pretrained_hero": PRETRAINED_HERO,
    "num_ppo_slots": NUM_PPO_SLOTS,
    "learning_rate": 5e-5,
    "n_steps": 4096,
    "batch_size": BATCH_SIZE,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef": 0.03,
    "features_dim": 99
}




def plot_training_results(log_dir, save_dir, num_timesteps):
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


def build_opponents_for_phase(phase: int, model_pool: List[str]) -> List[str]:
    """
    Construit la liste de 5 adversaires pour une phase donnée.
    
    - Slots 0-1 : rule + rule_v2 (fixes)
    - Slots 2-4 : PPO tirés aléatoirement du pool
    
    Phase 0 → 3× INITIAL_PPO_OPPONENT
    Phase 1+ → 3× random.choices(pool) avec remplacement
    """
    if phase == 0:
        ppo_opponents = [INITIAL_PPO_OPPONENT] * NUM_PPO_SLOTS
    else:
        ppo_opponents = random.choices(model_pool, k=NUM_PPO_SLOTS)
    
    opponents = FIXED_OPPONENTS + ppo_opponents
    return opponents


def make_env_factory(opponents_config: List[str]):
    """Retourne une factory function pour créer des envs avec une config d'adversaires spécifique."""
    def _make_env():
        try:
            env = PokerSB3Wrapper(
                num_opponents=len(opponents_config), 
                opponents_config=opponents_config
            )
            return env
        except Exception as e:
            print(f"⚠️ Erreur création env : {e}")
            return PokerSB3Wrapper(
                num_opponents=len(opponents_config), 
                opponents_config=['random'] * len(opponents_config)
            )
    return _make_env


def load_or_create_model(model_path: str, vec_env, tensorboard_log_dir: str, from_scratch: bool = False):
    """Charge un modèle pré-entraîné ou en crée un nouveau."""
    
    custom_objects = {
        "learning_rate": 5e-5,
        "clip_range": 0.15,
        "ent_coef": 0.03,
        "n_epochs": 10,
        "target_kl": 0.03
    }
    
    if not from_scratch and os.path.exists(model_path):
        print(f"♻️  Chargement du modèle : {model_path}")
        model = MaskablePPO.load(
            model_path,
            env=vec_env,
            tensorboard_log=tensorboard_log_dir,
            custom_objects=custom_objects,
            print_system_info=False
        )
    else:
        print("Création d'un modèle from scratch")
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=1,
        )
    
    return model



if __name__ == '__main__':
    
    print("=" * 70)
    print("   SELF-PLAY TRAINING WITH OPPONENT ROTATION")
    print("=" * 70)
    print(f"   Phases         : {NUM_PHASES} × {PHASE_STEPS:,} = {PHASE_STEPS * NUM_PHASES:,} total")
    print(f"   CPU            : {N_CPU}")
    print(f"   Fixed opponents: {FIXED_OPPONENTS}")
    print(f"   PPO slots      : {NUM_PPO_SLOTS} (rotated from pool)")
    print(f"   Initial PPO    : {INITIAL_PPO_OPPONENT}")
    print(f"   Hero pretrained: {PRETRAINED_HERO}")
    print("=" * 70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # ── W&B init (un seul run pour toutes les phases) ──
    run = None
    if os.getenv("WANDB_API_KEY"):
        print(f"🌊 Initialisation Weights & Biases...")
        run = wandb.init(
            project="poker",
            entity="killian-guillaume-personal",
            name=RUN_NAME,
            config=TRAINING_CONFIG,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
    else:
        print("⚠️ W&B désactivé (Pas de clé trouvée).")
    
    tensorboard_log_dir = f"{LOG_DIR}/tensorboard/"
    
    model_pool = [INITIAL_PPO_OPPONENT]
    if SEED_MODEL != INITIAL_PPO_OPPONENT and os.path.exists(SEED_MODEL):
        model_pool.append(SEED_MODEL)
    
    current_hero_path = PRETRAINED_HERO
    
    total_steps_done = 0
    
    try:
        for phase in range(NUM_PHASES):
            print(f"\n{'━' * 70}")
            print(f"🔄 PHASE {phase + 1}/{NUM_PHASES}  |  Steps: {total_steps_done:,} → {total_steps_done + PHASE_STEPS:,}")
            print(f"{'━' * 70}")
            
            opponents = build_opponents_for_phase(phase, model_pool)
            
            print(f"   🎯 Adversaires :")
            for i, opp in enumerate(opponents):
                if opp.endswith('.zip'):
                    label = os.path.basename(opp)
                else:
                    label = opp
                print(f"      Slot {i+1}: {label}")
            
            # 2. Créer l'environnement vectorisé
            make_env_fn = make_env_factory(opponents)
            vec_env = make_vec_env(make_env_fn, n_envs=N_CPU, vec_env_cls=SubprocVecEnv)
            vec_env = VecMonitor(vec_env, filename=f"{LOG_DIR}/monitor_phase{phase+1}.csv")
            
            # 3. Charger le modèle
            model = load_or_create_model(
                current_hero_path, 
                vec_env, 
                tensorboard_log_dir,
                from_scratch=(not os.path.exists(current_hero_path))
            )
            
            # 4. Callbacks
            smart_callback = SmartCheckpointCallback(
                save_freq=max(100_000 // N_CPU, 1),
                save_dir=SAVE_DIR,
                config={**TRAINING_CONFIG, "phase": phase + 1, "opponents": [os.path.basename(o) if o.endswith('.zip') else o for o in opponents]}
            )
            
            plotting_callback = PlottingCallback(
                log_dir=LOG_DIR,
                save_dir=SAVE_DIR,
                plot_func=plot_training_results,
                plot_freq=5000,
                verbose=0
            )
            
            callbacks_list = [smart_callback, plotting_callback]
            
            if run is not None:
                wandb_callback = WandbCallback(
                    gradient_save_freq=1000,
                    model_save_path=None,  # On gère la sauvegarde nous-mêmes
                    verbose=0,
                )
                callbacks_list.append(wandb_callback)
                
                # Log de la phase dans W&B
                wandb.log({
                    "phase": phase + 1,
                    "pool_size": len(model_pool),
                }, step=total_steps_done)
            
            # 5. Entraîner pour PHASE_STEPS
            print(f"\n🚀 Lancement phase {phase + 1} ({PHASE_STEPS:,} steps)...")
            model.learn(
                total_timesteps=PHASE_STEPS,
                callback=callbacks_list,
                log_interval=1,
                progress_bar=True,
                reset_num_timesteps=True,
                tb_log_name=f"{RUN_NAME}_phase{phase+1}"
            )
            
            # 6. Sauvegarder le checkpoint de cette phase
            phase_checkpoint = os.path.join(SAVE_DIR, f"phase_{phase + 1}.zip")
            model.save(phase_checkpoint)
            save_sb3_with_version(
                model, 
                f"PHASE_{phase+1}", 
                SAVE_DIR, 
                f"phase{phase+1}_{PHASE_STEPS}", 
                {**TRAINING_CONFIG, "phase": phase + 1}
            )
            print(f"💾 Checkpoint sauvegardé : {phase_checkpoint}")
            
            # 7. Ajouter ce checkpoint au pool d'adversaires
            model_pool.append(phase_checkpoint)
            print(f"📦 Pool de modèles : {len(model_pool)} modèles")
            for j, m in enumerate(model_pool):
                print(f"   [{j}] {os.path.basename(m)}")
            
            # 8. Mettre à jour le path Hero pour la prochaine phase
            current_hero_path = phase_checkpoint
            total_steps_done += PHASE_STEPS
            
            # 9. Fermer l'environnement (on le recrée à la phase suivante)
            vec_env.close()
            print(f"✅ Phase {phase + 1} terminée. Total: {total_steps_done:,} steps")
        
        # ── Sauvegarde finale ──
        print(f"\n{'═' * 70}")
        print(f"🏆 SELF-PLAY TERMINÉ — {total_steps_done:,} steps")
        print(f"{'═' * 70}")
        
        # Copier le dernier checkpoint comme modèle final
        final_path = os.path.join(SAVE_DIR, "FINAL_MODEL_SELF_PLAY.zip")
        import shutil
        shutil.copy2(current_hero_path, final_path)
        print(f"💾 Modèle final : {final_path}")
        
        # Sauvegarder le pool complet
        pool_info = {
            "model_pool": model_pool,
            "num_phases": NUM_PHASES,
            "phase_steps": PHASE_STEPS,
            "total_steps": total_steps_done,
            "config": TRAINING_CONFIG
        }
        with open(os.path.join(SAVE_DIR, "pool_info.json"), "w") as f:
            json.dump(pool_info, f, indent=2)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Interrompu à la phase {phase + 1}, step ~{total_steps_done:,}")
        interrupted_path = os.path.join(SAVE_DIR, f"INTERRUPTED_phase{phase+1}.zip")
        model.save(interrupted_path)
        print(f"💾 Sauvegardé : {interrupted_path}")
        
    finally:
        try:
            vec_env.close()
        except:
            pass
        if run is not None:
            wandb.finish()
        time.sleep(3)
        plot_training_results(LOG_DIR, SAVE_DIR, num_timesteps=total_steps_done)