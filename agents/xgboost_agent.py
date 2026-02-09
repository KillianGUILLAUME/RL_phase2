"""
Agent XGBoost qui joue dans RLCard
"""
from datetime import datetime


import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple
import pprint

from core.game_state import GameState
from adapters.rlcard_adapter import RLCardAdapter
from features.feature_builder import FeatureExtractor

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostRLCardAgent:
    """
    Agent qui utilise un modèle XGBoost entraîné sur Pluribus
    pour jouer dans RLCard
    """
    
    def __init__(self, model_path: str, env, use_safe_mode: bool = True):
        """
        Args:
            model_path: Chemin vers le modèle XGBoost (.json)
            use_safe_mode: Si True, ne raise jamais au call/fold si erreur
        """

        self.last_update = "En attente..."

        self.model = xgb.Booster()
        model_path = Path(model_path)
        if model_path.suffix == '.pkl':
            with open(model_path, 'rb') as f:
                import pickle
                self.model = pickle.load(f)
            
            # Vérifier le type
            if isinstance(self.model, xgb.Booster):
                logger.info("Modèle = xgb.Booster")
            elif isinstance(self.model, xgb.XGBClassifier):
                logger.info(" Modèle = xgb.XGBClassifier (sklearn API)")
                # Extraire le Booster interne
                self.model = self.model.get_booster()
            else:
                raise ValueError(f"Type de modèle non supporté: {type(self.model)}")
        
        elif model_path.suffix in ['.json', '.ubj']:
            # Chargement natif XGBoost
            logger.info(f"Chargement du modèle natif XGBoost...")
            # self.model = xgb.Booster({'nthread': 1}) 
            
            try:
                self.model.load_model(str(model_path))
                logger.info("✅ Modèle chargé avec succès (Single Thread mode)")
            except Exception as e:
                logger.error(f"CRASH au chargement: {e}")
                raise e
            
            logger.info("Modèle = xgb.Booster")
        
        else:
            raise ValueError(f"Format non supporté: {model_path.suffix}. Utilisez .pkl, .json ou .ubj")
        
        self.adapter = RLCardAdapter()
        self.extractor = FeatureExtractor()
        self.use_safe_mode = use_safe_mode
        self.use_raw = True

        self.env = env
        
        # Stats
        self.stats = {
            'total_decisions': 0,
            'errors': 0,
            'actions': {0: 0, 1: 0, 2: 0, 3: 0},
            'action_names': {0: 'FOLD', 1: 'CHECK/CALL', 2: 'RAISE', 3: 'ALL-IN'},
            'number_of_fold': 0
        }
        
        print(f"✅ XGBoost agent chargé depuis {model_path}")
    
    
    def step(self, state: Dict) -> int:
        """Appelé en mode training"""
        
        
        try:
            # === 1. Extraire legal actions ===
            legal_actions_enum = state.get('raw_legal_actions', [])
            legal_actions_int = [a.value for a in legal_actions_enum]
            
            if not legal_actions_int:
                logger.error(f"❌ State sans legal_actions! Keys: {state.keys()}")
                raise ValueError("State invalide!")
            
            # === 2. RLCard → GameState ===
            game_state = self.adapter.to_game_state(state, self.env)
            
            # === 3. GameState → Features ===
            X = self.extractor.extract(game_state)
            X_dmatrix = xgb.DMatrix(X.reshape(1, -1))
            
            # === 4. Prédiction ===
            proba = self.model.predict(X_dmatrix)[0]
            
            # === 5. Mapper vers action RLCard ===
            action_int, _ = self._map_to_rlcard_action(proba, legal_actions_int, game_state)

            action_enum = legal_actions_enum[legal_actions_int.index(action_int)]
            
            self.stats['actions'][action_int] += 1
            
            return action_enum
            
        except Exception as e:
            logger.error(f"❌ Erreur step: {e}", exc_info=True)
            self.stats['errors'] += 1
            fallback = legal_actions_enum[0]
            return fallback, {a: (1.0 if a == fallback else 0.0) for a in legal_actions_enum}

    
    
    def _map_to_rlcard_action(self, proba: np.ndarray, legal_actions: List[int], state: Dict = None) -> Tuple[int, Dict]:
        """
        Convertit les probas XGBoost avec stratégie de raise contextuelle
        """
        
        p_fold, p_call, p_raise = proba
        if np.argmax(proba) == 0:
            self.stats['number_of_fold'] += 1
        
        xgb_to_rlcard = {
        0: 0,  # FOLD → FOLD
        1: 1,  # CALL → CHECK/CALL
        2: 2   # RAISE → RAISE (par défaut)
        }
        
        # === 3. Gérer le cas où RAISE n'est pas disponible ===
        if 2 not in legal_actions and 3 in legal_actions:
            # RAISE pas dispo mais ALL-IN oui
            # Décision: ALL-IN seulement si forte conviction
            if p_raise > 0.7:
                xgb_to_rlcard[2] = 3  # RAISE → ALL-IN (si forte conviction)
                logger.debug(f"🎰 RAISE non dispo, p_raise={p_raise:.2f} > 0.7 → ALL-IN")
            else:
                xgb_to_rlcard[2] = 1  # RAISE → CALL (si conviction moyenne)
                logger.debug(f"🛑 RAISE non dispo, p_raise={p_raise:.2f} ≤ 0.7 → CALL")
        
        elif 2 not in legal_actions and 3 not in legal_actions:
            # Ni RAISE ni ALL-IN dispo (rare, mais possible)
            xgb_to_rlcard[2] = 1  # RAISE → CALL
            logger.debug(f"🚫 Ni RAISE ni ALL-IN dispo → CALL")
        
        # === 4. Construire les probas RLCard ===
        rlcard_probs = {a: 0.0 for a in legal_actions}
        
        # Redistribuer les probas XGBoost
        for xgb_class, p in enumerate([p_fold, p_call, p_raise]):
            rlcard_action = xgb_to_rlcard[xgb_class]
            
            # Si l'action est légale, ajouter la proba
            if rlcard_action in legal_actions:
                rlcard_probs[rlcard_action] += p
            else:
                # Sinon, redistribuer sur CALL (ou première action légale)
                fallback = 1 if 1 in legal_actions else legal_actions[0]
                rlcard_probs[fallback] += p
                logger.debug(f"⚠️ Action {rlcard_action} non légale, proba redistribuée sur {fallback}")
        
        # === 5. Normaliser (au cas où) ===
        total = sum(rlcard_probs.values())
        if total > 0:
            rlcard_probs = {a: p/total for a, p in rlcard_probs.items()}
        
        # === 6. Choisir l'action finale ===
        action = max(rlcard_probs.items(), key=lambda x: x[1])[0]
        
        # === 7. Validation ===
        if action not in legal_actions:
            logger.error(f"❌ Action {action} pas dans legal_actions {legal_actions}!")
            action = legal_actions[0]
            rlcard_probs = {a: (1.0 if a == action else 0.0) for a in legal_actions}
        
        return action, rlcard_probs

    
    def _safe_fallback(self, legal_actions: List[int]) -> Tuple[int, Dict]:
        """
        Mode sécurisé: call si possible, sinon fold
        """
        if 1 in legal_actions:  # call/check
            action = 1
        else:
            action = legal_actions[0]  # fold généralement
        
        probs_dict = {a: 1.0 if a == action else 0.0 for a in legal_actions}
        return action, probs_dict
    
    
    def eval_step(self, state: Dict) -> Tuple[int, Dict[int, float]]:
        """Appelé en mode évaluation"""
        # ✅ Extraire legal actions (gère OrderedDict + Enum + int)
        legal_actions_enum = state.get('raw_legal_actions', [])
        legal_actions_int = [a.value for a in legal_actions_enum]
        
        if not legal_actions_int:
            logger.error(f"❌ State sans legal_actions! Keys: {state.keys()}")
            raise ValueError("State invalide!")
        
        self.stats['total_decisions'] += 1
        
        try:
            # === 1. RLCard → GameState ===
            game_state = self.adapter.to_game_state(state, self.env)
            
            # === 2. GameState → Features ===
            X = self.extractor.extract(game_state)

            X_dmatrix = xgb.DMatrix(X.reshape(1, -1))
            
            # === 3. Prédiction XGBoost ===
            proba = self.model.predict(X_dmatrix)[0]
            
            # === 4. Mapper vers action RLCard ===
            action_int, probs_dict = self._map_to_rlcard_action(proba, legal_actions_int, game_state)

            action_enum = legal_actions_enum[legal_actions_int.index(action_int)]
            
            self.stats['actions'][action_int] += 1
            
            # ✅ Retourner l'ENUM, pas l'int !
            return action_enum, probs_dict
            
        except Exception as e:
            logger.error(f"❌ Erreur eval_step: {e}", exc_info=True)
            self.stats['errors'] += 1
            fallback = legal_actions_enum[0]
            return fallback, {a: (1.0 if a == fallback else 0.0) for a in legal_actions_enum}


    def _get_legal_actions_as_int(self, state: Dict) -> List[int]:
        """
        Extrait legal_actions du state RLCard et les convertit en int
        
        Gère les 3 formats possibles:
        1. OrderedDict {0: None, 1: None, ...} → [0, 1, ...]
        2. Liste d'Enum [<Action.FOLD: 0>, ...] → [0, 1, ...]
        3. Liste d'int [0, 1, ...] → [0, 1, ...]
        """
        import pprint
        pprint.pprint(state)
        # Essayer plusieurs clés possibles
        legal_actions = None
        
        # Priorité 1: 'raw_legal_actions' (liste d'Enum)
        if 'raw_legal_actions' in state:
            legal_actions = state['raw_legal_actions']
        
        # Priorité 2: 'legal_actions' (peut être OrderedDict ou liste)
        elif 'legal_actions' in state:
            legal_actions = state['legal_actions']
        
        # Priorité 3: Chercher dans 'raw_obs'
        elif 'raw_obs' in state and 'legal_actions' in state['raw_obs']:
            legal_actions = state['raw_obs']['legal_actions']
        
        else:
            logger.error(f"❌ Aucune legal_action trouvée dans state keys: {state.keys()}")
            return []
        
        # Cas 1: C'est un OrderedDict → extraire les keys
        if isinstance(legal_actions, dict):
            return list(legal_actions.keys())
        
        # Cas 2: C'est une liste
        if not legal_actions:
            logger.warning("⚠️ Liste legal_actions vide!")
            return []
        
        first = legal_actions[0]
        
        # Cas 2a: Liste d'Enum Action
        if hasattr(first, 'value'):
            return [int(a.value) for a in legal_actions]
        
        # Cas 2b: Liste d'int
        if isinstance(first, int):
            return legal_actions
        
        # Cas 2c: Autre type numérique (np.int64, etc.)
        try:
            return [int(a) for a in legal_actions]
        except Exception as e:
            logger.error(f"❌ Type inconnu: {type(first)} = {first}")
            raise ValueError(f"Impossible de convertir legal_actions: {e}")

    
    def get_stats(self) -> Dict:
        """Obtenir les statistiques de l'agent"""
        total = sum(self.stats['actions'].values())
        
        if total == 0:
            return self.stats
        
        # Ajouter les pourcentages
        action_percentages = {
            name: (count / total * 100)
            for action_id, count in self.stats['actions'].items()
            for name in [self.stats['action_names'][action_id]]
        }
        
        return {
            **self.stats,
            'action_percentages': action_percentages,
            'error_rate': self.stats['errors'] / self.stats['total_decisions'] * 100
        }
    
    def print_stats(self):
        """Afficher les statistiques"""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("📊 STATISTIQUES AGENT XGBOOST")
        print("="*50)
        print(f"Décisions totales: {stats['total_decisions']}")
        print(f"Erreurs: {stats['errors']} ({stats.get('error_rate', 0):.1f}%)")
        print("\nDistribution des actions:")
        
        for action_id, count in stats['actions'].items():
            name = stats['action_names'][action_id]
            pct = stats.get('action_percentages', {}).get(name, 0)
            print(f"  {name:12s}: {count:4d} ({pct:5.1f}%)")

        print(f"\nNombre de FOLD: {stats['number_of_fold']}")
        
        print("="*50 + "\n")
