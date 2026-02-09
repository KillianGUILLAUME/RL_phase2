import os
import time
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Imports RL
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Tes imports
from training.sb3wrapper import PokerSB3Wrapper

# ==============================================================================
# ⚙️ CONFIG TEST
# ==============================================================================
N_CPU = os.cpu_count()
TOTAL_TIMESTEPS = 2000     # Court pour le test
LOG_DIR = "./logs_test_plot"
SAVE_DIR = "./models_test_plot"

# Config factice pour tester le JSON
TEST_CONFIG = {
    "algo": "MaskablePPO",
    "description": "Crash Test Plotting & Saving",
    "learning_rate": 3e-4
}

# ==============================================================================
# 💾 1. SYSTÈME DE SAUVEGARDE (JSON + ZIP)
# ==============================================================================
def save_sb3_with_version(model, base_name, save_dir, step, config, metrics=None):
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarde ZIP (Poids)
    filename = f"{base_name}_{step}_steps"
    path = os.path.join(save_dir, filename)
    model.save(path)
    
    # Sauvegarde JSON (Métadonnées)
    metadata = {
        "step": step,
        "timestamp": time.time(),
        "config": config,
        "metrics": metrics or {}
    }
    json_path = os.path.join(save_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ [SAVE] Checkpoint créé : {filename}.json")

# ==============================================================================
# 📞 2. CALLBACK INTELLIGENT
# ==============================================================================
class SmartCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_dir, config, verbose=1):
        super(SmartCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.config = config

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            metrics_snapshot = {
                "n_calls": self.n_calls,
                "num_timesteps": self.num_timesteps
            }
            save_sb3_with_version(
                model=self.model,
                base_name="test_bot",
                save_dir=self.save_dir,
                step=self.num_timesteps,
                config=self.config,
                metrics=metrics_snapshot
            )
        return True

# ==============================================================================
# 📊 3. FONCTION DE PLOTTING (CORRIGÉE & ROBUSTE)
# ==============================================================================
def plot_test_results(log_dir):
    print(f"\n🎨 Génération du graphique depuis {log_dir}...")
    
    # Trouver le fichier events tfevents
    search_path = os.path.join(log_dir, "**", "events.out.tfevents*")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("❌ Aucun log Tensorboard trouvé ! (Attends que le writer flush)")
        return

    latest_file = max(files, key=os.path.getmtime)
    print(f"   Lecture de : {os.path.basename(latest_file)}")
    
    try:
        event_acc = EventAccumulator(latest_file)
        event_acc.Reload()
        
        # On extrait la reward
        tag = 'rollout/ep_rew_mean'
        if tag in event_acc.Tags()['scalars']:
            # --- CORRECTION ICI ---
            # On extrait les valeurs attribut par attribut au lieu de zipper
            events = event_acc.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.figure(figsize=(10, 5))
            plt.plot(steps, values, label='Reward Moyenne', color='blue', marker='o')
            plt.title("TEST : Courbe d'apprentissage (Reward)")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.legend()
            
            output_img = "test_plot_result.png"
            plt.savefig(output_img)
            print(f"✅ Graphique sauvegardé sous : {output_img}")
        else:
            print("⚠️ Pas assez de données pour tracer la courbe (ep_rew_mean manquant).")
            
    except Exception as e:
        print(f"❌ Erreur lors du plotting : {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 🚀 4. EXÉCUTION DU TEST
# ==============================================================================
def make_env():
    return PokerSB3Wrapper(num_opponents=5, str_opponents='random')

if __name__ == '__main__':
    print(f"🔥 TEST GLOBAL (Save + Plot) sur {N_CPU} cœurs...")
    
    # Nettoyage pour le test
    import shutil
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    if os.path.exists(SAVE_DIR): shutil.rmtree(SAVE_DIR)
    
    # 1. Env & Modèle
    vec_env = make_vec_env(make_env, n_envs=N_CPU, vec_env_cls=SubprocVecEnv)
    
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=128, 
        batch_size=64
    )
    
    # 2. Callback
    callback = SmartCheckpointCallback(
        save_freq=500, # Sauvegarde fréquente pour le test
        save_dir=SAVE_DIR,
        config=TEST_CONFIG
    )
    
    # 3. Learn
    print("🔄 Apprentissage en cours...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)
    
    # 4. Check Save Finale
    save_sb3_with_version(model, "FINAL_TEST", SAVE_DIR, "end", TEST_CONFIG)
    
    # 5. Check Plotting
    vec_env.close() 
    time.sleep(2) # On attend l'écriture disque
    
    plot_test_results(LOG_DIR)
    
    print("\n🎉 TEST COMPLET TERMINÉ !")
    print(f"1. Vérifie le dossier '{SAVE_DIR}' : Tu dois voir des .zip et .json")
    print(f"2. Vérifie l'image 'test_plot_result.png' : Tu dois voir une courbe.")