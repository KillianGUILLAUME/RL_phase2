import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rlcard
from rlcard.agents import RandomAgent
from features.feature_builder import FeatureExtractor_v2
from adapters.rlcard_adapter import RLCardAdapter
from agents.xgboost_agent import XGBoostRLCardAgent
from agents.onnx_policy import ONNXPokerBot
from sb3_contrib import MaskablePPO
from training.reward import PokerRewardShaper

import torch
torch.set_num_threads(1)

def dummy_schedule(progress_remaining: float) -> float:
    return 0.0003

    
class PPOBotAgent:
    """
    Transforme un modèle SB3 (.zip) en Agent compatible RLCard.
    OPTIMISATION : On ne garde que la 'Policy' (le cerveau) pour éviter
    les erreurs de pickle multiprocessing avec les lambdas du modèle complet.
    
    Compatibilité : auto-détecte la dimension du modèle chargé et
    tronque/pad l'observation si nécessaire (ex: modèle 87-feat vs pipeline 91-feat).
    """
    def __init__(self, model_path, env):
        self.env = env
        self.use_raw = True  # Requis par RLCard (env.run / tournament)
        
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
        self.policy = temp_model.policy
        self.policy.set_training_mode(False)
        
        # 3. Auto-détection de la dimension attendue par le modèle
        self.model_input_dim = self.policy.observation_space.shape[0]
        
        # Déduction du mode (99 ou 203 features)
        use_one_hot = self.model_input_dim > 99
        self.extractor = FeatureExtractor_v2(use_one_hot=use_one_hot)
        self.current_features_dim = self.extractor.NUM_FEATURES
        
        if self.model_input_dim != self.current_features_dim:
            print(f"⚠️ PPOBotAgent: modèle attend {self.model_input_dim} features, "
                  f"pipeline produit {self.current_features_dim}. Adaptation automatique.")
        
        del temp_model

    def step(self, state):
        # 1. Conversion State RLCard -> GameState
        try:
            game_state = RLCardAdapter.to_game_state(state, self.env)
            obs = self.extractor.extract(game_state)
            
            # Compatibilité : adapter la dimension si le modèle est ancien
            if self.model_input_dim < len(obs):
                obs = obs[:self.model_input_dim]  # Tronquer (nouvelles features à la fin)
            elif self.model_input_dim > len(obs):
                obs = np.pad(obs, (0, self.model_input_dim - len(obs)))  # Pad avec des 0
                
        except Exception as e: 
            raw_legal = state.get('raw_legal_actions', [])
            return raw_legal[0] if raw_legal else list(state['legal_actions'].keys())[0]

        # 2. Masques d'actions légales
        masks = np.zeros(self.env.num_actions, dtype=bool)
        legal_actions = list(state['legal_actions'].keys())
        for a in legal_actions:
            masks[a.value if hasattr(a, 'value') else int(a)] = True

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.policy.device).unsqueeze(0)
        masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.policy.device).unsqueeze(0)
        
        with torch.no_grad():
            action_tensor = self.policy._predict(obs_tensor, deterministic=True, action_masks=masks_tensor)
        action_idx = action_tensor.item()
        # 4. Convertir l'index en Action enum (requis par RLCard avec use_raw=True)
        raw_legal = state.get('raw_legal_actions', [])
        for a in raw_legal:
            if a.value == action_idx:
                return a
        
        # Fallback : première action légale
        return raw_legal[0] if raw_legal else action_idx

    def eval_step(self, state):
        action = self.step(state)
        return action, {}


class PokerSB3Wrapper(gym.Env):
    def __init__(self, num_opponents=5, opponents_config='xgb', use_one_hot=True):
        super(PokerSB3Wrapper, self).__init__()
        
        self.env = rlcard.make('no-limit-holdem', config={'game_num_players': num_opponents + 1})
        
        self.use_one_hot = use_one_hot
        self.extractor = FeatureExtractor_v2(use_one_hot=self.use_one_hot)
        self.num_opponents = num_opponents
        
        if isinstance(opponents_config, str):
            configs = [opponents_config] * num_opponents
        elif isinstance(opponents_config, list):
            if len(opponents_config) != num_opponents:
                raise ValueError(f"La liste doit contenir exactement {num_opponents} adversaires.")
            configs = opponents_config
        else:
            raise ValueError("opponents_config doit être une string ou une liste.")

        self.opponents = []
        for i, cfg in enumerate(configs):
            if cfg == 'random':
                self.opponents.append(RandomAgent(num_actions=self.env.num_actions))
                
            elif cfg == 'xgb':
                XGB_MODEL_PATH = 'models/xgb_87_features/xgb_pluribus_2026-01-29_12-31_fe6426.json'
                self.opponents.append(XGBoostRLCardAgent(model_path=XGB_MODEL_PATH, env=self.env))
                
            elif cfg == 'rule':
                from agents.rule_agents import RuleBasedBot
                self.opponents.append(RuleBasedBot(env=self.env))
                
            elif cfg == 'rule_v2':
                from agents.rule_agents import AdvancedRuleBot
                self.opponents.append(AdvancedRuleBot(env=self.env))
                
            elif cfg == 'psro':
                from agents.psro_agent import PSROAgent
                self.opponents.append(PSROAgent(env=self.env))

            elif cfg.endswith('.zip'):
                print(f"⚔️ Siège {i+1} : Chargement du modèle {cfg}")
                if os.path.exists(cfg):
                    self.opponents.append(PPOBotAgent(model_path=cfg, env=self.env))
                else:
                    print(f"⚠️ Fichier introuvable ({cfg}). Fallback -> RandomAgent.")
                    self.opponents.append(RandomAgent(num_actions=self.env.num_actions))
            elif cfg.endswith('.onnx'):
                print(f"⚡ Siège {i+1} : Chargement du modèle ONNX {cfg}")
                if os.path.exists(cfg):
                    self.opponents.append(ONNXPokerBot(onnx_model_path=cfg, env=self.env))
                else:
                    print(f"⚠️ Fichier ONNX introuvable ({cfg}). Fallback -> RandomAgent.")
                    self.opponents.append(RandomAgent(num_actions=self.env.num_actions))
            else:
                print(f"❓ Configuration inconnue ({cfg}). Fallback -> RandomAgent.")
                self.opponents.append(RandomAgent(num_actions=self.env.num_actions))
        
        dummy_hero = RandomAgent(num_actions=self.env.num_actions)
        all_agents = [dummy_hero] + self.opponents
        self.env.set_agents(all_agents)

        self.action_space = spaces.Discrete(self.env.num_actions)
        
        self.feature_extractor = self.extractor 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.extractor.NUM_FEATURES,), dtype=np.float32)

        self.reward_shaper = PokerRewardShaper(scale_factor=100.0)

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
        
        raw_reward = self.env.get_payoffs()[0] if done else 0
        reward = self.reward_shaper.shape(raw_reward) if done else 0.0
        
        if done:
            obs = np.zeros(self.observation_space.shape)
        else:
            game_state = RLCardAdapter.to_game_state(state, self.env)
            obs = self.extractor.extract(game_state)
        return np.array(obs, dtype=np.float32), reward, done, False, {}