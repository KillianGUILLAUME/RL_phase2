"""
Callbacks pour monitorer et contrôler l'entraînement.
Inspiré de Keras/PyTorch Lightning.
"""

import os

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt


class Callback(ABC):
    """Classe de base pour les callbacks."""
    
    def on_training_start(self, trainer):
        """Appelé au début de l'entraînement."""
        pass
    
    def on_training_end(self, trainer):
        """Appelé à la fin de l'entraînement."""
        pass
    
    def on_episode_start(self, trainer, episode: int):
        """Appelé au début de chaque épisode."""
        pass
    
    def on_episode_end(self, trainer, episode: int, metrics: Dict[str, Any]):
        """Appelé à la fin de chaque épisode."""
        pass
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        """Appelé après une évaluation."""
        pass
    
    def on_checkpoint_save(self, trainer, episode: int, save_path: str):
        """Appelé après sauvegarde d'un checkpoint."""
        pass


class ProgressCallback(Callback):
    """Affiche la progression de l'entraînement."""
    
    def __init__(self, print_every: int = 250):
        self.print_every = print_every
        self.start_time = None
        self.episode_times = []
    
    def on_training_start(self, trainer):
        self.start_time = time.time()
        print("🏋️‍♂️ Début de l'entraînement")
        print(f"   Episodes totaux : {trainer.config.training.num_episodes}")
        print(f"   Device : {trainer.device}")
        print(f"   Adversaires : {trainer.config.opponent.num_opponents}x {trainer.config.opponent.type}")
        print()
    
    def on_episode_end(self, trainer, episode: int, metrics: Dict[str, Any]):
        if episode % self.print_every == 0 and episode > 0:
            elapsed = time.time() - self.start_time
            eps_per_sec = episode / elapsed if elapsed > 0 else 0
            remaining = (trainer.config.training.num_episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"   Episode {episode}/{trainer.config.training.num_episodes} "
                  f"({eps_per_sec:.1f} ep/s, ~{remaining/60:.0f}min restantes)")
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        print(f"\n📈 Évaluation - Episode {episode}")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
        print()
    
    def on_training_end(self, trainer):
        elapsed = time.time() - self.start_time
        print(f"\n✅ Entraînement terminé en {elapsed/60:.1f} minutes")


class MetricsCallback(Callback):
    """Enregistre les métriques d'entraînement."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'win_rates': [],
            'epsilon': [],
            'loss': [],

            'eval_episodes': [],
            'eval_rewards': [],
            'eval_win_rates': [],
            'eval_avg_stack': []
        }
    
    def on_episode_end(self, trainer, **kwargs):
        episode = kwargs.get('episode', 0)
        metrics = kwargs.get('metrics', {})
        
        self.history['episodes'].append(episode)
        self.history['rewards'].append(metrics.get('reward', 0))
        self.history['avg_rewards'].append(metrics.get('avg_reward', 0))
        self.history['win_rates'].append(metrics.get('win_rate', 0))
        self.history['epsilon'].append(metrics.get('epsilon', 0))
        self.history['loss'].append(metrics.get('loss', 0))
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        self.history['eval_episodes'].append(episode)
        self.history['eval_rewards'].append(eval_metrics.get('avg_reward', 0))
    
    def on_training_end(self, trainer, **kwargs):
        """Sauvegarde finale des métriques."""
        metrics_file = self.save_dir / 'metrics.json'
        
        # 🆕 Conversion en types JSON-compatibles
        cleaned_history = self._convert_to_serializable(self.history)
        
        with open(metrics_file, 'w') as f:
            json.dump(cleaned_history, f, indent=2)
        
        print(f"📊 Métriques sauvegardées : {metrics_file}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Convertit récursivement les types NumPy/PyTorch en types Python natifs.
        
        Args:
            obj: Objet à convertir (dict, list, np.int64, etc.)
        
        Returns:
            Objet JSON-sérialisable
        """
        if isinstance(obj, dict):
            return {k: MetricsCallback._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [MetricsCallback._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Génère des graphiques de progression."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Récompenses d'entraînement (moyenne mobile)
        if len(self.history['rewards']) > 0:
            window = min(100, len(self.history['rewards']) // 10)
            if window > 1:
                smoothed = np.convolve(self.history['rewards'], 
                                      np.ones(window)/window, mode='valid')
                axes[0, 0].plot(smoothed, alpha=0.8)
            axes[0, 0].plot(self.history['rewards'], alpha=0.3, label='Raw')
            axes[0, 0].set_title('Récompenses d\'entraînement')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Récompenses d'évaluation
        if len(self.history['eval_rewards']) > 0:
            axes[0, 1].plot(self.history['eval_episodes'], 
                           self.history['eval_rewards'], 'o-', linewidth=2)
            axes[0, 1].set_title('Récompenses d\'évaluation')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution des récompenses
        if len(self.history['rewards']) > 0:
            axes[1, 0].hist(self.history['rewards'], bins=50, alpha=0.7)
            axes[1, 0].set_title('Distribution des récompenses')
            axes[1, 0].set_xlabel('Reward')
            axes[1, 0].set_ylabel('Fréquence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Progression temporelle
        if len(self.history['eval_rewards']) > 1:
            improvements = np.diff(self.history['eval_rewards'])
            axes[1, 1].plot(improvements, 'g-', alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Amélioration entre évaluations')
            axes[1, 1].set_xlabel('Évaluation #')
            axes[1, 1].set_ylabel('Δ Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Graphiques sauvegardés : {save_path}")
        else:
            plt.show()


class EarlyStoppingCallback(Callback):
    """Arrête l'entraînement si pas d'amélioration."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = -np.inf
        self.wait = 0
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        current_reward = eval_metrics.get('avg_reward', -np.inf)
        
        if current_reward > self.best_reward + self.min_delta:
            self.best_reward = current_reward
            self.wait = 0
            print(f"   🎯 Nouveau meilleur score : {self.best_reward:.4f}")
        else:
            self.wait += 1
            print(f"   ⏳ Pas d'amélioration ({self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                print(f"\n⚠️  Early stopping déclenché après {self.patience} évaluations sans amélioration")
                trainer.stop_training = True


class CheckpointCallback(Callback):
    """Gère la sauvegarde des checkpoints."""
    
    def __init__(self, save_best_only: bool = False):
        self.save_best_only = save_best_only
        self.best_reward = -np.inf
    
    def on_checkpoint_save(self, trainer, episode: int, save_path: str):
        print(f"💾 Checkpoint sauvegardé : {save_path}")
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        if self.save_best_only:
            current_reward = eval_metrics.get('avg_reward', -np.inf)
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                save_path = Path(trainer.config.training.save_dir) / trainer.config.training.experiment_name / 'best_model'
                save_path.mkdir(parents=True, exist_ok=True)
                trainer.agent.save_checkpoint(str(save_path))
                print(f"   💎 Meilleur modèle sauvegardé : {save_path}")


class TensorBoardCallback(Callback):
    """Log les métriques dans TensorBoard."""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.writer = None
    
    def on_training_start(self, trainer):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(self.log_dir) / trainer.config.training.experiment_name
            self.writer = SummaryWriter(str(log_path))
            print(f"📊 TensorBoard activé : {log_path}")
            print(f"   Lancer avec : tensorboard --logdir={self.log_dir}")
        except ImportError:
            print("⚠️  TensorBoard non disponible, pip install tensorboard")
    
    def on_episode_end(self, trainer, episode: int, metrics: Dict[str, Any]):
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, episode)
    
    def on_evaluation_end(self, trainer, episode: int, eval_metrics: Dict[str, Any]):
        if self.writer:
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'eval/{key}', value, episode)
    
    def on_training_end(self, trainer):
        if self.writer:
            self.writer.close()



def save_sb3_with_version(model, base_name, save_dir, step, config, metrics=None):
    """
    Sauvegarde le modèle SB3 (.zip) ET un fichier de métadonnées (.json).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Sauvegarde du modèle (poids)
    filename = f"{base_name}_{step}_steps"
    path = os.path.join(save_dir, filename)
    model.save(path)
    
    # 2. Sauvegarde des métadonnées (Config + Métriques)
    metadata = {
        "step": step,
        "timestamp": time.time(),
        "config": config,
        "metrics": metrics or {}
    }
    
    json_path = os.path.join(save_dir, f"{filename}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Checkpoint sauvegardé : {filename} (+ json)")


from stable_baselines3.common.callbacks import BaseCallback

class SmartCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_dir, config, verbose=1):
        super(SmartCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.config = config

    def _on_step(self) -> bool:
        # Cette fonction est appelée à chaque pas de temps
        
        # On vérifie si c'est le moment de sauvegarder
        if self.n_calls % self.save_freq == 0:
            
            current_epsilon = 0
            if hasattr(self.model, 'exploration_rate'):
                current_epsilon = self.model.exploration_rate

            metrics_snapshot = {
                "epsilon": current_epsilon,
                "n_calls": self.n_calls,
                "num_timesteps": self.num_timesteps
            }
            
            save_sb3_with_version(
                model=self.model,
                base_name="ppo_sb3_poker",
                save_dir=self.save_dir,
                step=self.num_timesteps,
                config=self.config,
                metrics=metrics_snapshot
            )
            
        return True