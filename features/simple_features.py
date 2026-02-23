from typing import List
import numpy as np
from core.game_state import GameState

class SimpleFeature:

    POSITION_ORDER = ['SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN']
    STREETS = ['preflop', 'flop', 'turn', 'river']

    # ═══════════════════════════════════════════════════════════
    # 2. FEATURES DE POSITION (6)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_position_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        # Position normalisée (1)
        if state.position in self.POSITION_ORDER:
            out[idx] = self.POSITION_ORDER.index(state.position) / (len(self.POSITION_ORDER) - 1)
        else:
            out[idx] = 0.5  # Position inconnue = milieu
        
        # Distance au bouton (1)
        if state.position == 'BTN':
            distance = 0.0
        elif state.position == 'CO':
            distance = 1.0 / 7.0
        elif state.position in ['MP', 'MP+1']:
            distance = 3.0 / 7.0
        elif state.position in ['UTG', 'UTG+1']:
            distance = 5.0 / 7.0
        elif state.position == 'BB':
            distance = 6.0 / 7.0
        elif state.position == 'SB':
            distance = 7.0 / 7.0
        else:
            distance = 0.5
        out[idx + 1] = distance
        
        # In position (1)
        out[idx + 2] = 1.0 if state.position in ['BTN', 'CO'] else 0.0
        
        # Position one-hot: early, middle, late (3)
        out[idx + 3] = 1.0 if state.position in ['SB', 'BB', 'UTG', 'UTG+1'] else 0.0
        out[idx + 4] = 1.0 if state.position in ['MP', 'MP+1'] else 0.0
        out[idx + 5] = 1.0 if state.position in ['CO', 'BTN'] else 0.0
        
        return idx + 6  # 6 features
    
    # ═══════════════════════════════════════════════════════════
    # 3. FEATURES DE STACK & POT (12)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_stack_pot_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        stack_bb = state.effective_stack_bb
        spr = state.spr
        
        out[idx] = min(stack_bb / 200.0, 1.0)
        out[idx + 1] = min(state.pot_size_bb / 100.0, 1.0)
        out[idx + 2] = min(spr / 20.0, 1.0)
        out[idx + 3] = state.pot_odds
        out[idx + 4] = min(state.amount_to_call_bb / 50.0, 1.0)
        out[idx + 5] = 1.0 if state.is_all_in_situation else 0.0
        
        out[idx + 6] = 1.0 if stack_bb < 20 else 0.0
        out[idx + 7] = 1.0 if 20 <= stack_bb < 100 else 0.0
        out[idx + 8] = 1.0 if stack_bb >= 100 else 0.0
        
        out[idx + 9] = 1.0 if spr < 4 else 0.0
        out[idx + 10] = 1.0 if 4 <= spr < 13 else 0.0
        out[idx + 11] = 1.0 if spr >= 13 else 0.0
        
        return idx + 12  # 12 features
    
    # ═══════════════════════════════════════════════════════════
    # 4. FEATURES D'ACTIONS (15)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_action_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        actions = state.actions_this_street
        
        out[idx] = min(len(actions) / 10.0, 1.0)
        
        num_folds = sum(1 for a in actions if 'fold' in a.lower())
        num_calls = sum(1 for a in actions if 'call' in a.lower())
        num_raises = sum(1 for a in actions if 'raise' in a.lower())
        num_checks = sum(1 for a in actions if 'check' in a.lower())
        num_bets = sum(1 for a in actions if 'bet' in a.lower())
        num_allin = sum(1 for a in actions if 'allin' in a.lower())
        
        passive_actions = num_calls + num_checks
        aggressive_actions = num_bets + num_raises
        out[idx + 1] = aggressive_actions / max(passive_actions + aggressive_actions, 1)
        
        out[idx + 2] = min(num_folds / 3.0, 1.0)
        out[idx + 3] = min(num_calls / 3.0, 1.0)
        out[idx + 4] = min(num_raises / 3.0, 1.0)
        out[idx + 5] = min(num_checks / 3.0, 1.0)
        out[idx + 6] = min(num_bets / 3.0, 1.0)
        out[idx + 7] = min(num_allin / 2.0, 1.0)
        
        last_aggressive = 0.0
        if actions:
            last_action = actions[-1].lower()
            if any(x in last_action for x in ['bet', 'raise', 'allin']):
                last_aggressive = 1.0
        out[idx + 8] = last_aggressive
        
        last_aggression = state.get_last_aggression_amount()
        last_aggression_bb = (last_aggression / state.big_blind) if state.big_blind > 0 else 0
        out[idx + 9] = min(last_aggression_bb / 50.0, 1.0)
        
        out[idx + 10] = 1.0 if state.amount_to_call > 0 else 0.0
        out[idx + 11] = 1.0 if 'fold' in state.legal_actions else 0.0
        out[idx + 12] = 1.0 if ('check' in state.legal_actions or 'call' in state.legal_actions) else 0.0
        out[idx + 13] = 1.0 if 'raise' in state.legal_actions else 0.0
        out[idx + 14] = 1.0 if ('all_in' in state.legal_actions or 'allin' in state.legal_actions) else 0.0
        
        return idx + 15  # 15 features
    
    # ═══════════════════════════════════════════════════════════
    # 5. FEATURES DE CONTEXTE (12)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_context_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        num_players = state.num_active_players
        out[idx] = num_players / 9.0  # Normalisation (max 9 joueurs)
        
        for i, street in enumerate(self.STREETS):
            out[idx + 1 + i] = 1.0 if state.street == street else 0.0
            
        bb_sb_ratio = state.big_blind / max(state.small_blind, 1)
        out[idx + 5] = min(bb_sb_ratio / 3.0, 1.0)
        
        # Nouvelles features d'historique (Remplacent les `players_x` one-hot redondants)
        out[idx + 6] = 1.0 if state.is_single_raised_pot else 0.0
        out[idx + 7] = 1.0 if state.is_3bet_pot else 0.0
        out[idx + 8] = 1.0 if state.was_multiway_flop else 0.0
        
        # On garde quelques features de base sur la taille de la table pour combler jusqu'à 12 features
        # (Pour maintenir exactement nos 12 features de la section Contexte et donc 203 au total)
        out[idx + 9] = 1.0 if num_players == 2 else 0.0
        out[idx + 10] = 1.0 if num_players in [3, 4] else 0.0
        out[idx + 11] = 1.0 if num_players >= 5 else 0.0
        
        return idx + 12  # 12 features
    