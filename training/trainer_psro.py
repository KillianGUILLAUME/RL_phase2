import os
import shutil
import glob
import time
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import wandb
from wandb.integration.sb3 import WandbCallback
from training.callbacks import SmartCheckpointCallback, PlottingCallback
from dotenv import load_dotenv

load_dotenv()

from training.sb3wrapper import PokerSB3Wrapper
from training.psro_evaluator import evaluate_and_solve

# ==========================================
# ⚙️ POLICY SPACE RESPONSE ORACLES (PSRO) - GENERATIONAL LOOP
# ==========================================
ARCHIVE_DIR = "models/psro_archive"
BASE_MODELS_DIR = "models/psro_runs"

GENERATIONS = 50                 
STEPS_PER_GENERATION = 3_000_000 

INITIAL_SEED_MODEL = "models/psro_archive/psro_training_1500000_steps.zip"

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
        fig.suptitle(f'PSRO Generation — {num_timesteps} steps', fontsize=16)
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

def setup_archive():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    existing_models = glob.glob(os.path.join(ARCHIVE_DIR, "*.zip"))
    
    if len(existing_models) == 0:
        print(f"📦 Archive PSRO est vide. Insertion du modèle initial...")
        if os.path.exists(INITIAL_SEED_MODEL):
            dest = os.path.join(ARCHIVE_DIR, "gen_00_champion.zip")
            shutil.copy(INITIAL_SEED_MODEL, dest)
            print(f"✅ Modèle d'origine inséré: {dest}")
        else:
            raise FileNotFoundError(f"Impossible de trouver le modèle racine {INITIAL_SEED_MODEL}.")
    else:
        print(f"📦 {len(existing_models)} modèles trouvés dans l'Archive.")

def make_env():
    """Factory isolée pour le SubprocVecEnv s'attaquant au Méta-Agent PSRO en 6-Max."""
    def _init():
        return PokerSB3Wrapper(num_opponents=5, opponents_config="psro")
    return _init

def train_oracles():
    setup_archive()
    
    while True:
        # ÉTAPE 1: Mathématiques! Calculer la Matrice des Gains et le Nash Equilibrium.
        # Cela lit l'archive, fait jouer les NOUVEAUX modèles contre les ANCIENS, 
        # actualise la Matrice et enregistre les Nash Weights !
        print(f"\n{'='*60}")
        print("🧠 PHASE 1: EVALUATEUR EMPIRIQUE & SOLVEUR NASH")
        print(f"{'='*60}")
        evaluate_and_solve()
        
        # Combien de modèles ont été créés jusqu'ici ?
        existing_models = len(glob.glob(os.path.join(ARCHIVE_DIR, "*.zip")))
        # Si on a généré gen_00, c'est que la prochaine best response à créer est la gen_01
        gen_id = existing_models
        gen_str = f"{gen_id:03d}"
        
        if gen_id > GENERATIONS:
            print(f"🎉 Les {GENERATIONS} itérations PSRO sont terminées ! Le GTO est atteint.")
            break
            
        print(f"\n{'='*60}")
        print(f"💪 PHASE 2: FABRICATION DE LA MEILLEURE REPONSE (GÉNÉRATION {gen_id}/{GENERATIONS})")
        print(f"{'='*60}")
        
        # ÉTAPE 2: Entraînement de l'Oracle (Le PPO qui casse le Méta-Agent actuel)
        model_save_dir = os.path.join(BASE_MODELS_DIR, f"generation_{gen_str}")
        os.makedirs(model_save_dir, exist_ok=True)
        
        num_envs = max(1, os.cpu_count() // 2)
        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
        env = VecMonitor(env)
        
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            verbose=1,
            tensorboard_log=model_save_dir,
            policy_kwargs=dict(net_arch=[512, 256, 128]),
            device="cpu"
        )
        
        # Configuration for WandB and Saving
        training_config = {
            "algo": "MaskablePPO_PSRO",
            "generation": gen_id,
            "total_timesteps": STEPS_PER_GENERATION,
            "learning_rate": 5e-5,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.02,
        }
        
        smart_callback = SmartCheckpointCallback(
            save_freq=max(500_000 // num_envs, 1), 
            save_dir=model_save_dir, 
            config=training_config
        )

        plotting_callback = PlottingCallback(
            log_dir=model_save_dir,
            save_dir=model_save_dir,
            plot_func=plot_training_results,
            plot_freq=500_000,
            verbose=0
        )

        callbacks_list = [smart_callback, plotting_callback]

        if os.getenv("WANDB_API_KEY"):
            print(f"🌊 Initialisation Weights & Biases pour la Génération {gen_id}...")
            run = wandb.init(
                project="poker",
                entity="killian-guillaume-personal",
                name=f"PSRO_Gen_{gen_str}",
                config=training_config,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
                reinit=True
            )
            wandb_callback = WandbCallback(
                gradient_save_freq=1000,
                model_save_path=None,
                verbose=0,
            )
            callbacks_list.append(wandb_callback)
        else:
            print("⚠️ W&B désactivé (Pas de clé trouvée).")
            
        model.learn(
            total_timesteps=STEPS_PER_GENERATION,
            callback=callbacks_list,
            progress_bar=True
        )
        
        if os.getenv("WANDB_API_KEY"):
            wandb.finish()
        
        final_path = os.path.join(model_save_dir, f"gen_{gen_str}_final.zip")
        model.save(final_path)
        
        # Archiver le nouveau modèle
        archive_path = os.path.join(ARCHIVE_DIR, f"gen_{gen_str}.zip")
        shutil.copy(final_path, archive_path)
        print(f"📚 Nouvelle stratégie archivée: {os.path.basename(archive_path)}")
        
        env.close()

if __name__ == '__main__':
    train_oracles()
