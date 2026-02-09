import gym
from gym import spaces
import numpy as np

import rlcard
from rlcard.envs import Env
from adapters.rlcard_adapter import RLCardAdapter
from features.feature_builder import FeatureExtractor

class PPOAgent(gym.Env):
    def __init__(self, env, model_path=None, state_shape=None):
        super(PPOAgent, self).__init__()
        self.env = env
        self.adapter = RLCardAdapter()
        self.extractor = FeatureExtractor()

        if state_shape is None:
            # On fait une extraction à blanc pour avoir la taille
            dummy_state = self._get_dummy_state()
            state_shape = [len(dummy_state['obs'])]

        # Définir les espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.extractor.extract(self._get_dummy_state())),),  # Taille du vecteur de features
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.env.num_actions)  # Actions discrètes (fold/call/raise)

    def _get_dummy_state(self):
        """Helper pour calculer la taille des features au démarrage"""
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

    def reset(self):
        state, _ = self.env.reset()
        return self._process_state(state)

    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        processed_state = self._process_state(next_state)
        return processed_state, reward, done, info

    def _process_state(self, state):
        game_state = self.adapter.to_game_state(state, self.env)
        features = self.extractor.extract(game_state)
        return {
            'obs': np.array(features, dtype=np.float32),
            'legal_actions': state.get('legal_actions', {}),
            'raw_legal_actions': state.get('raw_legal_actions', [])
        }

    def render(self, mode='human'):
        pass  # Optionnel : pour afficher l'état du jeu


class LegalActionWrapper(gym.Wrapper):
    def __init__(self, env, model=None):
        super(LegalActionWrapper, self).__init__(env)
        self.model = model 

    def step(self, action):
        legal_actions = self.env.env.get_legal_actions()

        if action not in legal_actions:
            if self.model is not None:
                obs = self.env._get_obs()

                action_probs = self.model.policy.get_distribution(obs).distribution.probs
                action_probs = action_probs.detach().numpy().flatten()

                masked_probs = np.zeros_like(action_probs)
                for a in legal_actions:
                    masked_probs[a] = action_probs[a]

                if np.sum(masked_probs) > 0:
                    masked_probs /= np.sum(masked_probs)
                else:
                    masked_probs = np.zeros_like(action_probs)
                    for a in legal_actions:
                        masked_probs[a] = 1.0 / len(legal_actions)
                action = np.argmax(masked_probs)
            else:
                action = np.random.choice(legal_actions)

        return self.env.step(action)

