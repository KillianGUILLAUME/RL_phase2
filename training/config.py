"""
Configuration pour l'entraînement d'agents RL au poker.
Utilise dataclasses pour une configuration type-safe et flexible.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import torch
import json
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration de l'environnement de poker."""
    game: str = 'no-limit-holdem'
    seed: int = 42
    num_players: int = 6
    
    def to_dict(self) -> Dict[str, Any]:
        return {'seed': self.seed}


@dataclass
class OpponentConfig:
    """Configuration des adversaires."""
    type: str = 'xgboost'  # 'random', 'xgboost', 'rule_based', 'self_play', etc.
    model_path: Optional[str] = None
    num_opponents: int = 5
    
    # Pour self-play
    use_self_play: bool = False
    self_play_ratio: float = 0.5  # % d'adversaires en self-play


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement."""
    num_episodes: int = 10000
    eval_every: int = 1000
    eval_episodes: int = 200
    save_every: int = 2500
    print_every: int = 250
    
    # Checkpointing
    save_dir: str = 'models/rl'
    experiment_name: str = 'experiment'
    resume_from: Optional[str] = None
    
    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # Multi-GPU (pour plus tard)
    use_multi_gpu: bool = False
    
    def get_device(self) -> torch.device:
        """Retourne le device approprié."""
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class LoggingConfig:
    """Configuration du logging et monitoring."""
    log_dir: str = 'logs'
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    # Métriques à tracker
    track_metrics: List[str] = field(default_factory=lambda: [
        'reward', 'epsilon', 'loss', 'q_values', 'win_rate'
    ])
    
    # Verbosity
    verbose: int = 1  # 0: silent, 1: normal, 2: debug


@dataclass
class FullTrainingConfig:
    """Configuration complète regroupant tous les aspects."""
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Paramètres spécifiques à l'agent (flexibles)
    agent_params: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str):
        """Sauvegarde la config en JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"💾 Configuration sauvegardée : {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FullTrainingConfig':
        """Charge une config depuis JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'env': asdict(self.env),
            'opponent': asdict(self.opponent),
            'training': asdict(self.training),
            'logging': asdict(self.logging),
            'agent_params': self.agent_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FullTrainingConfig':
        """Crée depuis un dictionnaire."""
        return cls(
            env=EnvironmentConfig(**data.get('env', {})),
            opponent=OpponentConfig(**data.get('opponent', {})),
            training=TrainingConfig(**data.get('training', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            agent_params=data.get('agent_params', {})
        )


# 🎯 Configs pré-définies pour différents scénarios

def get_quick_test_config() -> FullTrainingConfig:
    """Config pour tests rapides (5 min)."""
    return FullTrainingConfig(
        training=TrainingConfig(
            num_episodes=1000,
            eval_every=250,
            eval_episodes=50,
            save_every=500,
            print_every=100,
            experiment_name='quick_test'
        ),
        logging=LoggingConfig(verbose=2)
    )


def get_standard_training_config() -> FullTrainingConfig:
    """Config standard pour entraînement complet."""
    return FullTrainingConfig(
        training=TrainingConfig(
            num_episodes=50000,
            eval_every=2000,
            eval_episodes=500,
            save_every=5000,
            experiment_name='standard_training'
        )
    )


def get_kaggle_config() -> FullTrainingConfig:
    """Config optimisée pour Kaggle (12h max)."""
    return FullTrainingConfig(
        training=TrainingConfig(
            num_episodes=100000,
            eval_every=5000,
            eval_episodes=300,
            save_every=10000,
            device='cuda',
            experiment_name='kaggle_training'
        ),
        logging=LoggingConfig(
            tensorboard=False,  # Kaggle n'aime pas trop
            verbose=1
        )
    )


def get_self_play_config() -> FullTrainingConfig:
    """Config pour self-play."""
    return FullTrainingConfig(
        opponent=OpponentConfig(
            type='self_play',
            use_self_play=True,
            self_play_ratio=0.7,
            num_opponents=5
        ),
        training=TrainingConfig(
            num_episodes=100000,
            experiment_name='self_play'
        )
    )
