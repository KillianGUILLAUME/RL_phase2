import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def find_latest_log_dir(base_dir="./logs"):
    """Trouve le dossier de log le plus récent."""
    if not os.path.exists(base_dir):
        print(f"❌ Le dossier {base_dir} n'existe pas encore.")
        return None
        
    # On cherche tous les dossiers 'PPO_x' dans les sous-dossiers de logs
    # Structure typique : ./logs/run_123456/PPO_1/events...
    search_path = os.path.join(base_dir, "**", "events.out.tfevents*")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("❌ Aucun fichier de log trouvé.")
        return None
        
    # On prend le fichier le plus récent
    latest_file = max(files, key=os.path.getmtime)
    return os.path.dirname(latest_file)

def plot_training_summary(log_dir):
    print(f"📂 Lecture des logs depuis : {log_dir}")
    
    # Chargement des données TensorBoard
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extraction des métriques disponibles
    tags = event_acc.Tags()['scalars']
    
    # Configuration du plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Résumé Entraînement : {os.path.basename(os.path.dirname(log_dir))}', fontsize=16)
    
    # 1. Récompense Moyenne (Le plus important)
    if 'rollout/ep_rew_mean' in tags:
        steps, _, values = zip(*event_acc.Scalars('rollout/ep_rew_mean'))
        axs[0, 0].plot(steps, values, color='#2ecc71', linewidth=2)
        axs[0, 0].set_title('Récompense Moyenne (Performance)', fontweight='bold')
        axs[0, 0].set_xlabel('Steps')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True, alpha=0.3)
    else:
        axs[0, 0].text(0.5, 0.5, "Pas encore de données\n(Attend quelques épisodes)", ha='center')

    # 2. Entropy Loss (Exploration)
    if 'train/entropy_loss' in tags:
        steps, _, values = zip(*event_acc.Scalars('train/entropy_loss'))
        axs[0, 1].plot(steps, values, color='#e67e22')
        axs[0, 1].set_title('Entropie (Exploration)', fontweight='bold')
        axs[0, 1].set_xlabel('Steps')
        axs[0, 1].grid(True, alpha=0.3)
        
    # 3. Value Loss (Erreur de prédiction)
    if 'train/value_loss' in tags:
        steps, _, values = zip(*event_acc.Scalars('train/value_loss'))
        axs[1, 0].plot(steps, values, color='#e74c3c')
        axs[1, 0].set_title('Value Loss (Critique)', fontweight='bold')
        axs[1, 0].set_xlabel('Steps')
        axs[1, 0].grid(True, alpha=0.3)

    # 4. Longueur des épisodes
    if 'rollout/ep_len_mean' in tags:
        steps, _, values = zip(*event_acc.Scalars('rollout/ep_len_mean'))
        axs[1, 1].plot(steps, values, color='#3498db')
        axs[1, 1].set_title('Durée Moyenne des Parties', fontweight='bold')
        axs[1, 1].set_xlabel('Steps')
        axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Sauvegarde
    output_file = "training_summary.png"
    plt.savefig(output_file, dpi=150)
    print(f"📊 Graphique généré : {output_file}")
    plt.show()

if __name__ == "__main__":
    latest_log = find_latest_log_dir()
    if latest_log:
        try:
            plot_training_summary(latest_log)
        except Exception as e:
            print(f"Erreur lors du plot : {e}")
            print("💡 Conseil : Installe tensorboard avec 'pip install tensorboard'")