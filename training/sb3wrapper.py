import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rlcard
from rlcard.agents import RandomAgent
from features.feature_builder import FeatureExtractor
from adapters.rlcard_adapter import RLCardAdapter
from agents.xgboost_agent import XGBoostRLCardAgent
from sb3_contrib import MaskablePPO


def dummy_schedule(progress_remaining: float) -> float:
    return 0.0003

    
# --- CLASSE PPOBotAgent (VERSION LÉGÈRE) ---
class PPOBotAgent:
    """
    Transforme un modèle SB3 (.zip) en Agent compatible RLCard.
    OPTIMISATION : On ne garde que la 'Policy' (le cerveau) pour éviter
    les erreurs de pickle multiprocessing avec les lambdas du modèle complet.
    """
    def __init__(self, model_path, env):
        self.env = env
        
        # 1. On charge le modèle complet temporairement
        try:
            temp_model = MaskablePPO.load(
                model_path, 
                custom_objects={
                    "learning_rate": dummy_schedule,
                    "lr_schedule": dummy_schedule,
                    "clip_range": dummy_schedule
                }
            )
        except Exception:
            temp_model = MaskablePPO.load(model_path)
            
        # 2. CHIRURGIE : On extrait juste le cerveau (Policy) 🧠
        # Cela supprime les optimiseurs et schedules qui font planter le pickle
        self.policy = temp_model.policy
        
        # On passe en mode évaluation (fige les poids)
        self.policy.set_training_mode(False)
        
        # 3. On supprime le gros modèle lourd
        del temp_model
            
        self.extractor = FeatureExtractor()

    def step(self, state):
        # 1. Conversion State RLCard -> GameState
        try:
            game_state = RLCardAdapter.to_game_state(state, self.env)
            obs = self.extractor.extract(game_state)
        except Exception:
            # Fallback random
            return np.random.choice(list(state['legal_actions'].keys()))

        # 2. Masques
        masks = np.zeros(self.env.num_actions, dtype=bool)
        legal_actions = list(state['legal_actions'].keys())
        masks[legal_actions] = True

        # 3. Prédiction avec la POLICY directement (et pas model.predict)
        # La policy MaskablePPO accepte bien l'argument action_masks
        action, _ = self.policy.predict(obs, action_masks=masks, deterministic=True)
        
        return action

    def eval_step(self, state):
        action = self.step(state)
        return action, {}


class PokerSB3Wrapper(gym.Env):
    def __init__(self, num_opponents=5, str_opponents = 'xgb'):
        super(PokerSB3Wrapper, self).__init__()
        
        self.env = rlcard.make('no-limit-holdem', config={'game_num_players': num_opponents + 1})
        
        self.extractor = FeatureExtractor()
        self.num_opponents = num_opponents
        
        if str_opponents == 'random':
            self.opponents = [RandomAgent(num_actions=self.env.num_actions) for _ in range(num_opponents)]
            dummy_hero = RandomAgent(num_actions=self.env.num_actions)
            all_agents = [dummy_hero] + self.opponents
            self.env.set_agents(all_agents)
        elif str_opponents == 'xgb':
            XGB_MODEL_PATH = 'models/xgb/xgb_pluribus_2026-01-29_12-31_fe6426.json'
            self.opponents = [XGBoostRLCardAgent(model_path=XGB_MODEL_PATH, env = self.env) for _ in range(num_opponents)]
            dummy_hero = RandomAgent(num_actions=self.env.num_actions)
            all_agents = [dummy_hero] + self.opponents
            self.env.set_agents(all_agents)
        elif str_opponents.endswith('.zip'): #we directly give the path to the model
            print(f"⚔️ SELF-PLAY : Chargement du modèle {str_opponents}")
            if os.path.exists(str_opponents):
                # On passe self.env à chaque agent
                self.opponents = [
                    PPOBotAgent(model_path=str_opponents, env=self.env) 
                    for _ in range(num_opponents)
                ]
                dummy_hero = RandomAgent(num_actions=self.env.num_actions)
                all_agents = [dummy_hero] + self.opponents
                self.env.set_agents(all_agents)
            else:
                raise FileNotFoundError(f"Model not found: {str_opponents}")
        else:
            print('pas d\'agent')
        self.action_space = spaces.Discrete(self.env.num_actions)
        
        self.feature_extractor = FeatureExtractor() 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(87,), dtype=np.float32)


    def _play_until_my_turn(self, state):
        """
        Cette méthode fait jouer tous les adversaires (XGBoost/Random)
        jusqu'à ce que ce soit à nouveau au tour du Joueur 0 (Nous) ou que la partie finisse.
        """
        current_player = state['raw_obs']['current_player']
        
        # Tant que ce n'est pas à nous (0) et que le jeu continue
        while (current_player != 0) and (not self.env.is_over()):
            # On récupère l'agent correspondant au joueur actuel
            action, _ = self.env.agents[current_player].eval_step(state)
            
            # On joue l'action dans l'environnement
            state, _ = self.env.step(action)
            
            # Mise à jour du joueur actuel
            current_player = state['raw_obs']['current_player']
            
        return state, self.env.is_over()

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state, _ = self.env.reset()
        state, done = self._play_until_my_turn(state)
        if done: return self.reset(seed=seed)
        
        # Conversion Dict -> GameState -> Numpy
        game_state = RLCardAdapter.to_game_state(state, self.env)
        obs = self.extractor.extract(game_state)
        
        return np.array(obs, dtype=np.float32), {}


    def action_masks(self):
        """
        Retourne un tableau booléen indiquant les actions valides.
        True = Action Autorisée
        False = Action Interdite
        """
        # On récupère l'état brut pour connaître les coups légaux
        # Note: get_player_id() est important pour s'assurer qu'on regarde le bon joueur
        raw_state = self.env.get_state(self.env.get_player_id())
        
        # RLCard renvoie un dict où les clés sont les indices d'actions légales
        # ex: {0: 'Fold', 1: 'Check'} -> keys sont [0, 1]
        legal_actions = list(raw_state['legal_actions'].keys())
        
        # On crée un masque rempli de FALSE (tout interdit par défaut)
        masks = np.zeros(self.action_space.n, dtype=bool)
        
        # On met à TRUE uniquement les actions légales
        masks[legal_actions] = True
        
        return masks

    def step(self, action):
        # On joue directement
        state, _ = self.env.step(action)
        state, done = self._play_until_my_turn(state)
        
        reward = self.env.get_payoffs()[0] if done else 0
        
        # Gestion de la fin d'épisode vs continuation
        if done:
            obs = np.zeros(self.observation_space.shape)
            # info = {'mask': ...} n'est pas obligatoire ici, MaskablePPO appelle action_masks() directement
        else:
            game_state = RLCardAdapter.to_game_state(state, self.env)
            obs = self.extractor.extract(game_state)
        
        return np.array(obs, dtype=np.float32), reward, done, False, {}