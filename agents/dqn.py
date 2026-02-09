import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.games.nolimitholdem.round import Action
from adapters.rlcard_adapter import RLCardAdapter
from features.feature_builder import FeatureExtractor

class SmartDQNAgent(DQNAgent):
    """
    Un Agent DQN qui utilise ton FeatureExtractor au lieu des bits bruts.
    """
    def __init__(
        self, 
        env, 
        model_path=None, 
        state_shape=None,
        device=None,
        # 🆕 Tous les paramètres DQN maintenant configurables
        mlp_layers=None,
        replay_memory_size=50000,
        batch_size=32,
        replay_memory_init_size=1000,
        update_target_estimator_every=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
        learning_rate=0.00005,
        discount_factor=0.99,
        **kwargs  # 🆕 Capture les paramètres non listés
    ):
        """
        Args:
            env: Environnement RLCard
            model_path: Chemin pour charger un modèle existant
            state_shape: Taille de l'état (auto-détecté si None)
            device: Device PyTorch ('cpu', 'cuda', ou None pour auto)
            mlp_layers: Architecture du réseau [256, 128] par défaut
            replay_memory_size: Taille de la mémoire de replay
            batch_size: Taille des batchs d'entraînement
            replay_memory_init_size: Taille minimum avant d'entraîner
            update_target_estimator_every: Fréquence de mise à jour du target network
            epsilon_start: Epsilon initial pour exploration
            epsilon_end: Epsilon final
            epsilon_decay_steps: Steps pour décroissance epsilon
            learning_rate: Learning rate de l'optimizer
            discount_factor: Gamma pour le discount des rewards
        """
        
        # 1. On charge tes outils magiques
        self.adapter = RLCardAdapter()
        self.extractor = FeatureExtractor()
        self.env = env
        
        # 2. Gestion du device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        print(f"🖥️  SmartDQNAgent sur device: {self.device}")

        # 3. Détection de state_shape si non fourni
        if state_shape is None:
            dummy_state = self._get_dummy_state()
            state_shape = [len(dummy_state['obs'])]
            print(f"🧠 SmartDQN: Input Layer détecté = {state_shape}")

        self.state_shape = state_shape
        
        # 4. Architecture par défaut
        if mlp_layers is None:
            mlp_layers = [256, 128]

        # 5. 🔑 Initialisation du parent DQNAgent avec TOUS les paramètres
        super().__init__(
            num_actions=env.num_actions,
            state_shape=state_shape,
            mlp_layers=mlp_layers,
            replay_memory_size=replay_memory_size,
            batch_size=batch_size,
            replay_memory_init_size=replay_memory_init_size,
            update_target_estimator_every=update_target_estimator_every,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            device=self.device,
            **kwargs  # Passe les paramètres supplémentaires non capturés
        )

        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._mlp_layers = mlp_layers
        self._replay_memory_size = replay_memory_size
        self._batch_size = batch_size
        self._replay_memory_init_size = replay_memory_init_size
        self._update_target_estimator_every = update_target_estimator_every
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps
        
        self._move_to_device()

        if model_path:
            self.load_checkpoint(model_path)

    def _move_to_device(self):
        """Déplace les réseaux sur self.device."""
        if hasattr(self, 'q_estimator') and hasattr(self.q_estimator, 'qnet'):
            self.q_estimator.qnet = self.q_estimator.qnet.to(self.device)
        
        if hasattr(self, 'target_estimator') and hasattr(self.target_estimator, 'qnet'):
            self.target_estimator.qnet = self.target_estimator.qnet.to(self.device)
    
    def save_checkpoint(self, path: str):
        import time
        """
        Sauvegarde device-agnostic.
        Surcharge la méthode parente pour gérer CPU/GPU.
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarde sur CPU pour compatibilité maximale
        checkpoint = {
            # ===== Réseaux de neurones =====
            'q_estimator': self.q_estimator.qnet.state_dict(),
            'target_estimator': self.target_estimator.qnet.state_dict(),
            
            # ===== Hyperparamètres d'architecture =====
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'mlp_layers': self._mlp_layers,  # ✅ Avec underscore
            
            # ===== Hyperparamètres d'entraînement =====
            'learning_rate': self._learning_rate,  # ✅ Avec underscore
            'discount_factor': self._discount_factor,  # ✅ Avec underscore
            'batch_size': self._batch_size,  # ✅ Avec underscore
            'replay_memory_size': self._replay_memory_size,  # ✅ Avec underscore
            'replay_memory_init_size': self._replay_memory_init_size,  # ✅ Avec underscore
            'update_target_estimator_every': self._update_target_estimator_every,  # ✅ Avec underscore
            
            # ===== Exploration =====
            'epsilon_start': self._epsilon_start,  # ✅ Avec underscore
            'epsilon_end': self._epsilon_end,  # ✅ Avec underscore
            'epsilon_decay_steps': self._epsilon_decay_steps,  # ✅ Avec underscore
            'epsilons': self.epsilons,
            'total_t': self.total_t,
            
            # ===== Métadonnées =====
            'timestamp': time.time(),
            'device': str(self.device)
        }

        
        save_path = os.path.join(path, 'checkpoint.pt')
        torch.save(checkpoint, save_path)
        
        # Remettre sur le device d'origine
        self._move_to_device()
        
        print(f"💾 Checkpoint sauvegardé : {save_path}")

        


    def load_checkpoint(self, path: str):
        """
        Chargement device-agnostic.
        Surcharge la méthode parente pour gérer CPU/GPU.
        """
        import os
        
        if os.path.isdir(path):
            checkpoint_path = os.path.join(path, 'checkpoint.pt')
        else:
            checkpoint_path = path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
        
        # 🔑 map_location : charge sur self.device peu importe d'où ça vient
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self._learning_rate = checkpoint.get('learning_rate', self._learning_rate)
        self._discount_factor = checkpoint.get('discount_factor', self._discount_factor)
        self._mlp_layers = checkpoint.get('mlp_layers', self._mlp_layers)
        
        self.q_estimator.qnet.load_state_dict(checkpoint['q_estimator'])
        self.target_estimator.qnet.load_state_dict(checkpoint['target_estimator'])
        
        # Restauration de l'exploration
        self.epsilon_start = checkpoint.get('epsilon_start', 1.0)
        self.epsilon_end = checkpoint.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = checkpoint.get('epsilon_decay_steps', 100000)
        self.epsilons = checkpoint.get('epsilons', [self.epsilon_start] * self.env.num_players)
        self.total_t = checkpoint.get('total_t', 0)
        
        avg_epsilon = sum(self.epsilons) / len(self.epsilons)
        print(f"✅ Checkpoint chargé : {checkpoint_path}")
        print(f"   Steps: {self.total_t}, ε_moyen: {avg_epsilon:.4f}")
        self._move_to_device()

        print(f"✅ Checkpoint chargé : {checkpoint_path} sur {self.device}")

    @property
    def learning_rate(self):
        """Extrait le learning rate depuis l'optimizer."""
        return self.q_estimator.optimizer.param_groups[0]['lr']

    # @property
    # def discount_factor(self):
    #     """Retourne le discount factor."""
    #     # RLCard stocke discount_factor directement
    #     return getattr(self, 'discount_factor', self._discount_factor)


    def _process_state(self, state):
        """
        C'est ICI la magie.
        On intercepte le 'state' brut de RLCard.
        On le transforme en 'features' intelligentes.
        On le renvoie au DQN.
        """
        game_state = self.adapter.to_game_state(state, self.env)
        
        features_vector = self.extractor.extract(game_state)
        
        return {
            'obs': np.array(features_vector, dtype=np.float32),
            'legal_actions': state.get('legal_actions', {}),
            'raw_legal_actions': state.get('raw_legal_actions', [])
        }

    def feed(self, ts):
        """
        Surcharge de la méthode d'apprentissage (Training).
        On doit convertir (state, action, next_state, reward, done).
        """
        (state, action, reward, next_state, done) = tuple(ts)
        
        smart_state = self._process_state(state)
        smart_next_state = self._process_state(next_state)
        
        super().feed((smart_state, action, reward, smart_next_state, done))

    # def step(self, state):
    #     smart_state_dict = self._process_state(state)
    #     return super().step(smart_state_dict)

    def step(self, state):
        """
        Surcharge de la prise de décision (Training Action).
        """
        smart_state_dict = self._process_state(state)
        legal_actions = smart_state_dict['legal_actions']

        if not legal_actions:
            return 0 

        obs = smart_state_dict['obs']

        obs_batch = np.expand_dims(obs, 0)

        q_values = self.q_estimator.predict_nograd(obs_batch)

        masked_q_values = np.where(
            np.isin(np.arange(len(q_values)), legal_actions),  
            q_values,
            -np.inf
        )

        best_action = np.argmax(masked_q_values)
        
        return best_action
    
    

    def eval_step(self, state):
        """
        Surcharge de l'évaluation (Playing Action).
        """
        smart_state_dict = self._process_state(state)
        return super().eval_step(smart_state_dict)

    def _get_dummy_state(self):
        """Helper pour calculer la taille des features"""
        # On crée un faux state RLCard minimaliste
        mock_rlcard_state = {
            'player_id': 0,
            'current_player': 0,
            'legal_actions': {0: None, 1: None, 2: None},
            'raw_legal_actions': [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT],
            'raw_obs': {
                'hand': ['Ah', 'Kh'], 
                'public_cards': ['Td', '8s', '2c'], 
                'pot': 100, 
                'all_chips': [1000]*6,
                'my_chips': 1000,
                'stakes': [1, 2, 0, 0, 0, 0],
                'current_player': 0,
                'player_id': 0,
                'legal_actions': [Action.FOLD, Action.CHECK_CALL]
            }
        }
        return self._process_state(mock_rlcard_state)
    
