"""
Trainer agnostique pour agents de poker RL.
Compatible DQN, PPO, A2C, SAC, etc.
"""

import numpy as np
import torch
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import reorganize
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

from .config import FullTrainingConfig
from .callbacks import Callback, ProgressCallback, MetricsCallback


class PokerRLTrainer:
    """
    Trainer universel pour agents RL au poker.
    
    Principe : Le trainer est agnostique de l'algorithme RL utilisé.
    Il suffit que l'agent implémente :
      - feed(transition) : pour les algos on-policy (DQN, etc.)
      - ou update(trajectories) : pour les algos off-policy (PPO, etc.)
      - eval_step(state) : pour l'évaluation
      - save_checkpoint(path) / load_checkpoint(path)
    """
    
    def __init__(
        self,
        agent,  # N'importe quel agent RL
        config: FullTrainingConfig,
        callbacks: Optional[List[Callback]] = None
    ):
        self.agent = agent
        self.config = config
        self.callbacks = callbacks or [ProgressCallback(), MetricsCallback()]
        
        # Setup environnement
        self.env = rlcard.make(
            self.config.env.game,
            config=self.config.env.to_dict()
        )
        
        # Setup device
        self.device = self.config.training.get_device()
        if hasattr(self.agent, 'device'):
            self.agent.device = self.device
        
        # Setup adversaires
        self.opponents = self._create_opponents()
        
        # Stats
        self.current_episode = 0
        self.stop_training = False
        
        print(f"🎮 Trainer initialisé")
        print(f"   Environnement : {self.config.env.game}")
        print(f"   Agent : {type(agent).__name__}")
        print(f"   Device : {self.device}")
    
    def _create_opponents(self) -> List:
        """Crée les agents adversaires selon la config."""
        opponents = []
        
        for i in range(self.config.opponent.num_opponents):
            if self.config.opponent.type == 'random':
                opponents.append(RandomAgent(self.env.num_actions))
            
            elif self.config.opponent.type == 'xgboost':
                from agents.xgboost_agent import XGBoostRLCardAgent
                opponents.append(
                    XGBoostRLCardAgent(
                        model_path=self.config.opponent.model_path,
                        env=self.env
                    )
                )
            
            elif self.config.opponent.type == 'self_play':
                # En self-play, on met des copies de notre agent
                # (on les mettra à jour périodiquement)
                opponents.append(self.agent)
            
            else:
                raise ValueError(f"Type d'adversaire inconnu : {self.config.opponent.type}")
        
        return opponents
    
    def _trigger_callbacks(self, event: str, **kwargs):
        """Déclenche un événement sur tous les callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(self, **kwargs)
    
    def evaluate(self, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        Évalue l'agent contre des adversaires aléatoires.
        Retourne un dict de métriques.
        """
        n_episodes = n_episodes or self.config.training.eval_episodes
        
        # Setup environnement d'évaluation avec adversaires random
        eval_agents = [self.agent] + [RandomAgent(self.env.num_actions) 
                                       for _ in range(self.env.num_players - 1)]
        self.env.set_agents(eval_agents)
        
        rewards = []
        wins = 0
        
        for _ in range(n_episodes):
            trajectories, payoffs = self.env.run(is_training=False)
            rewards.append(payoffs[0])
            if payoffs[0] > 0:
                wins += 1
        
        metrics = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'win_rate': wins / n_episodes,
        }
        
        return metrics
    
    def train_episode(self) -> Dict[str, Any]:
        """
        Exécute un épisode d'entraînement.
        Retourne les métriques de l'épisode.
        """
        # Setup agents pour l'entraînement
        training_agents = [self.agent] + self.opponents
        self.env.set_agents(training_agents)
        
        # Exécution
        trajectories, payoffs = self.env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        
        # Apprentissage (méthode agnostique)
        if hasattr(self.agent, 'feed'):
            # Style DQN : on feed transition par transition
            for ts in trajectories[0]:
                self.agent.feed(ts)
        
        elif hasattr(self.agent, 'update'):
            # Style PPO : on update avec toute la trajectoire
            self.agent.update(trajectories[0])
        
        else:
            raise NotImplementedError(
                "L'agent doit implémenter feed() ou update()"
            )
        
        # Métriques de l'épisode
        metrics = {
            'reward': payoffs[0],
        }
        
        # Ajouter des métriques spécifiques à l'agent si disponibles
        if hasattr(self.agent, 'get_metrics'):
            agent_metrics = self.agent.get_metrics()
            metrics.update(agent_metrics)
        
        return metrics
    
    def train(self):
        """Lance l'entraînement complet."""
        self._trigger_callbacks('on_training_start')
        
        try:
            for episode in range(self.config.training.num_episodes):
                if self.stop_training:
                    print("⚠️  Entraînement arrêté par callback")
                    break
                
                self.current_episode = episode
                
                # Episode d'entraînement
                self._trigger_callbacks('on_episode_start', episode=episode)
                metrics = self.train_episode()
                self._trigger_callbacks('on_episode_end', episode=episode, metrics=metrics)
                
                # Évaluation périodique
                if episode % self.config.training.eval_every == 0 and episode > 0:
                    eval_metrics = self.evaluate()
                    self._trigger_callbacks('on_evaluation_end', 
                                           episode=episode, 
                                           eval_metrics=eval_metrics)
                
                # Sauvegarde périodique
                if episode % self.config.training.save_every == 0 and episode > 0:
                    self.save_checkpoint(episode)
        
        finally:
            self._trigger_callbacks('on_training_end')
    
    def save_checkpoint(self, episode: Optional[int] = None):
        """Sauvegarde un checkpoint."""
        if episode is None:
            episode = self.current_episode
        
        save_dir = Path(self.config.training.save_dir) / self.config.training.experiment_name
        save_path = save_dir / f'checkpoint_{episode}'
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde de l'agent
        self.agent.save_checkpoint(str(save_path))
        
        # Sauvegarde de la config
        config_path = save_path / 'config.json'
        self.config.save(str(config_path))
        
        self._trigger_callbacks('on_checkpoint_save', 
                               episode=episode, 
                               save_path=str(save_path))
    
    def resume_training(self, checkpoint_path: str):
        """Reprend l'entraînement depuis un checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Charge l'agent
        if hasattr(self.agent, 'load_checkpoint'):
            self.agent.load_checkpoint(str(checkpoint_path))
            print(f"✅ Agent chargé depuis {checkpoint_path}")
        
        # TODO: Charger l'épisode actuel depuis metadata
