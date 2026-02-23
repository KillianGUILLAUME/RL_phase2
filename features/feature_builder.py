# features/feature_extractor.py

import numpy as np
from typing import Dict, List, Tuple
from core.game_state import GameState
import logging

from treys import Evaluator, Card

logger = logging.getLogger(__name__)

from .card_features   import CardFeatures
from .simple_features import SimpleFeature      
from .game_theory     import GameTheoryFeatures 



"""
IMPORTANT :
For each features we add, we have to put it at the end of the list of features.
Like this, we will be able to still use older models.
"""


class FeatureExtractor:
    """
    Extracteur de features universel pour le poker No-Limit Hold'em.
    
    Compatible avec:
    - Pluribus (via PluribusAdapter)
    - RLCard (via RLCardAdapter)
    
    Features extraites (99 au total):
    ═══════════════════════════════════════════════════════════
    1. CARTES (22 features)
       - Hand strength (preflop/postflop)
       - Equity estimée
       - Draws (flush, straight)
    
    2. POSITION (6 features)
       - Position relative (BTN=1.0, SB=0.0)
       - Distance au bouton
       - In/out of position
    
    3. STACK & POT (12 features)
       - Stack effectif en BB
       - SPR (Stack-to-Pot Ratio)
       - Pot odds
       - All-in situations
    
    4. ACTIONS (15 features)
       - Aggression factor
       - Nombre de bets/raises
       - Last aggressor
    
    5. CONTEXTE (12 features)
       - Nombre de joueurs actifs
       - Street (one-hot)
       - Blinds ratio
    
    6. THÉORIE DU JEU 
       - EV estimé (call/fold/raise)
       - Fold equity, Implied/Reverse implied odds
       - MDF, Alpha, Bet-to-pot ratio, Pot geometry
       - Range advantage, Board coverage, Polarisation
       - Betting patterns
    """
    
    # ═══════════════════════════════════════════════════════════
    # CONSTANTES
    # ═══════════════════════════════════════════════════════════
    
    # Positions ordonnées du moins avantageux au plus avantageux
    POSITION_ORDER = ['SB', 'BB', 'UTG', 'UTG+1', 'MP', 'MP+1', 'CO', 'BTN']
    
    # Ranges de mains preflop (simplifiés)
    PREMIUM_HANDS = {
        'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'
    }
    
    STRONG_HANDS = {
        'TT', '99', 'AQs', 'AQo', 'AJs', 'KQs'
    }
    
    # Streets one-hot
    STREETS = ['preflop', 'flop', 'turn', 'river']
    
    # Nombre total de features
    NUM_FEATURES = 99
    
    def __init__(self):
        """Initialise l'extracteur."""
        self.feature_names = self._generate_feature_names()
        self.evaluator = Evaluator()
        logger.info(f"FeatureExtractor initialisé avec {self.NUM_FEATURES} features")
    
    # ═══════════════════════════════════════════════════════════
    # EXTRACTION PRINCIPALE
    # ═══════════════════════════════════════════════════════════
    
    def extract(self, state: GameState) -> np.ndarray:
        """
        Extrait toutes les features d'un GameState.
        
        Args:
            state: GameState standardisé
        
        Returns:
            np.ndarray de shape (91,) avec toutes les features
        """


        features = []
        
        # 1. Features de cartes (22)
        features.extend(self._extract_card_features(state))
        
        # 2. Features de position (6)
        features.extend(self._extract_position_features(state))
        
        # 3. Features de stack & pot (12)
        features.extend(self._extract_stack_pot_features(state))
        
        # 4. Features d'actions (15)
        features.extend(self._extract_action_features(state))
        
        # 5. Features de contexte (12)
        features.extend(self._extract_context_features(state))
        
        # 6. Features de théorie du jeu (20)
        features.extend(self._extract_game_theory_features(state))
        
        # Validation
        features_array = np.array(features, dtype=np.float32)
        if len(features_array) != self.NUM_FEATURES:
            logger.error(
                f"Nombre de features incorrect: {len(features_array)} != {self.NUM_FEATURES}"
            )
            raise ValueError(f"Expected {self.NUM_FEATURES} features, got {len(features_array)}")
        
        return features_array
    
    # ═══════════════════════════════════════════════════════════
    # 1. FEATURES DE CARTES (22)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_card_features(self, state: GameState) -> List[float]:
        """
        Extrait les features liées aux cartes.
        
        Returns:
            Liste de 22 features:
            - [0-1]: Cartes encodées (rank1, rank2)
            - [2]: Suited (0 ou 1)
            - [3]: Pocket pair (0 ou 1)
            - [4-5]: Hand strength preflop (premium, strong)
            - [6]: Hand strength postflop (0-1)
            - [7]: Equity estimée (0-1)
            - [8-9]: Flush draw (made, draw)
            - [10-11]: Straight draw (made, draw)
            - [12]: Overcards au board
            - [13-16]: Pair/TwoPair/Trips/Quads
            - [17-21]: Board texture (coordinated, wet, etc.)
        """
        features = []
        
        hole_cards = state.hole_cards
        board = state.board
        
        # Encodage des cartes (2)
        if len(hole_cards) >= 2:
            rank1 = self._card_rank_to_value(hole_cards[0])
            rank2 = self._card_rank_to_value(hole_cards[1])
            features.extend([rank1 / 14.0, rank2 / 14.0])  # Normalisation
        else:
            features.extend([0.0, 0.0])
        
        # Suited (1)
        if len(hole_cards) >= 2:
            s1 = self._get_suit_char(hole_cards[0])
            s2 = self._get_suit_char(hole_cards[1])
            suited = 1.0 if s1 == s2 else 0.0
        else:
            suited = 0.0
        features.append(suited)
        
        # Pocket pair (1)
        if len(hole_cards) >= 2:
            r1 = self._get_rank_char(hole_cards[0])
            r2 = self._get_rank_char(hole_cards[1])
            pocket_pair = 1.0 if r1 == r2 else 0.0
        else:
            pocket_pair = 0.0
        features.append(pocket_pair)
        
        # Hand strength preflop (2)
        hand_str = self._get_preflop_hand_string(hole_cards)
        is_premium = 1.0 if hand_str in self.PREMIUM_HANDS else 0.0
        is_strong = 1.0 if hand_str in self.STRONG_HANDS else 0.0
        features.extend([is_premium, is_strong])
        
        # Hand strength postflop (1)
        if state.street != 'preflop' and len(board) >= 3:
            postflop_strength = self._evaluate_hand_strength(hole_cards, board)
        else:
            postflop_strength = 0.0
        features.append(postflop_strength)
        
        # Equity estimée (1)
        equity = self._estimate_equity(hole_cards, board, state.num_active_players)
        features.append(equity)
        
        # Flush draws (2)
        flush_made, flush_draw = self._check_flush(hole_cards, board)
        features.extend([flush_made, flush_draw])
        
        # Straight draws (2)
        straight_made, straight_draw = self._check_straight(hole_cards, board)
        features.extend([straight_made, straight_draw])
        
        # Overcards (1)
        overcards = self._count_overcards(hole_cards, board)
        features.append(overcards / 2.0)  # Normalisation (max 2)
        
        # Hand made (4): pair, two_pair, trips, quads
        pair, two_pair, trips, quads = self._check_hand_made(hole_cards, board)
        features.extend([pair, two_pair, trips, quads])
        
        # Board texture (5)
        if len(board) >= 3:
            texture = self._analyze_board_texture(board)
            features.extend([
                texture['coordinated'],
                texture['wet'],
                texture['paired'],
                texture['high_cards'],
                texture['monotone']
            ])
        else:
            features.extend([0.0] * 5)
        
        return features  # 22 features
    
    # ═══════════════════════════════════════════════════════════
    # 2. FEATURES DE POSITION (6)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_position_features(self, state: GameState) -> List[float]:
        """
        Extrait les features liées à la position.
        
        Returns:
            Liste de 6 features:
            - [0]: Position normalisée (0=SB, 1=BTN)
            - [1]: Distance au bouton
            - [2]: In position (0 ou 1)
            - [3-5]: Position one-hot (early/middle/late)
        """
        features = []
        
        # Position normalisée (1)
        if state.position in self.POSITION_ORDER:
            pos_value = self.POSITION_ORDER.index(state.position) / (len(self.POSITION_ORDER) - 1)
        else:
            pos_value = 0.5  # Position inconnue = milieu
        features.append(pos_value)
        
        # Distance au bouton (1)
        # BTN = 0, CO = 1, MP = 2, etc.
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
        features.append(distance)
        
        # In position (1)
        # Simplifié: BTN, CO sont toujours IP postflop
        in_position = 1.0 if state.position in ['BTN', 'CO'] else 0.0
        features.append(in_position)
        
        # Position one-hot: early, middle, late (3)
        early = 1.0 if state.position in ['SB', 'BB', 'UTG', 'UTG+1'] else 0.0
        middle = 1.0 if state.position in ['MP', 'MP+1'] else 0.0
        late = 1.0 if state.position in ['CO', 'BTN'] else 0.0
        features.extend([early, middle, late])
        
        return features  # 6 features
    
    # ═══════════════════════════════════════════════════════════
    # 3. FEATURES DE STACK & POT (12)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_stack_pot_features(self, state: GameState) -> List[float]:
        """
        Extrait les features liées au stack et au pot.
        
        Returns:
            Liste de 12 features:
            - [0]: Stack effectif (BB) normalisé
            - [1]: Pot size (BB) normalisé
            - [2]: SPR (Stack-to-Pot Ratio)
            - [3]: Pot odds
            - [4]: Amount to call (BB)
            - [5]: All-in situation (0 ou 1)
            - [6-8]: Stack size (short/medium/deep)
            - [9-11]: SPR category (low/medium/high)
        """
        features = []
        
        # Stack effectif en BB (1)
        stack_bb = state.effective_stack_bb
        stack_normalized = min(stack_bb / 200.0, 1.0)  # Cap à 200BB
        features.append(stack_normalized)
        
        # Pot size en BB (1)
        pot_bb = state.pot_size_bb
        pot_normalized = min(pot_bb / 100.0, 1.0)  # Cap à 100BB
        features.append(pot_normalized)
        
        # SPR (1)
        spr = state.spr
        spr_normalized = min(spr / 20.0, 1.0)  # Cap à 20
        features.append(spr_normalized)
        
        # Pot odds (1)
        pot_odds = state.pot_odds
        features.append(pot_odds)
        
        # Amount to call en BB (1)
        amount_to_call_bb = state.amount_to_call_bb
        amount_normalized = min(amount_to_call_bb / 50.0, 1.0)  # Cap à 50BB
        features.append(amount_normalized)
        
        # All-in situation (1)
        all_in = 1.0 if state.is_all_in_situation else 0.0
        features.append(all_in)
        
        # Stack categories (3): short, medium, deep
        short_stack = 1.0 if stack_bb < 20 else 0.0
        medium_stack = 1.0 if 20 <= stack_bb < 100 else 0.0
        deep_stack = 1.0 if stack_bb >= 100 else 0.0
        features.extend([short_stack, medium_stack, deep_stack])
        
        # SPR categories (3): low, medium, high
        low_spr = 1.0 if spr < 4 else 0.0
        medium_spr = 1.0 if 4 <= spr < 13 else 0.0
        high_spr = 1.0 if spr >= 13 else 0.0
        features.extend([low_spr, medium_spr, high_spr])
        
        return features  # 12 features
    
    # ═══════════════════════════════════════════════════════════
    # 4. FEATURES D'ACTIONS (15)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_action_features(self, state: GameState) -> List[float]:
        """
        Extrait les features liées aux actions.
        
        Returns:
            Liste de 15 features:
            - [0]: Nombre d'actions cette street
            - [1]: Aggression factor
            - [2-4]: Nombre de folds/calls/raises
            - [5-7]: Nombre de checks/bets/all-ins
            - [8]: Last action was aggressive (0 ou 1)
            - [9]: Last aggression amount (BB)
            - [10]: Facing bet/raise (0 ou 1)
            - [11-14]: Actions disponibles (fold/call/check/raise)
        """
        features = []
        
        actions = state.actions_this_street
        
        # Nombre d'actions (1)
        num_actions = len(actions)
        features.append(min(num_actions / 10.0, 1.0))  # Normalisation
        
        # Comptage des types d'actions (6)
        num_folds = sum(1 for a in actions if 'fold' in a.lower())
        num_calls = sum(1 for a in actions if 'call' in a.lower())
        num_raises = sum(1 for a in actions if 'raise' in a.lower())
        num_checks = sum(1 for a in actions if 'check' in a.lower())
        num_bets = sum(1 for a in actions if 'bet' in a.lower())
        num_allin = sum(1 for a in actions if 'allin' in a.lower())
        
        # Aggression factor (1)
        passive_actions = num_calls + num_checks
        aggressive_actions = num_bets + num_raises
        aggression = aggressive_actions / max(passive_actions + aggressive_actions, 1)
        features.append(aggression)
        
        # Compteurs normalisés (6)
        features.extend([
            min(num_folds / 3.0, 1.0),
            min(num_calls / 3.0, 1.0),
            min(num_raises / 3.0, 1.0),
            min(num_checks / 3.0, 1.0),
            min(num_bets / 3.0, 1.0),
            min(num_allin / 2.0, 1.0)
        ])
        
        # Last action was aggressive (1)
        last_aggressive = 0.0
        if actions:
            last_action = actions[-1].lower()
            if any(x in last_action for x in ['bet', 'raise', 'allin']):
                last_aggressive = 1.0
        features.append(last_aggressive)
        
        # Last aggression amount (1)
        last_aggression = state.get_last_aggression_amount()
        last_aggression_bb = (last_aggression / state.big_blind) if state.big_blind > 0 else 0
        features.append(min(last_aggression_bb / 50.0, 1.0))
        
        # Facing bet/raise (1)
        facing_aggression = 1.0 if state.amount_to_call > 0 else 0.0
        features.append(facing_aggression)
        
        can_fold = 1.0 if 'fold' in state.legal_actions else 0.0
        can_check_call = 1.0 if ('check' in state.legal_actions or 'call' in state.legal_actions) else 0.0
        can_raise = 1.0 if 'raise' in state.legal_actions else 0.0
        can_allin = 1.0 if ('all_in' in state.legal_actions or 'allin' in state.legal_actions) else 0.0
        
        features.extend([can_fold, can_check_call, can_raise, can_allin])
        
        return features  # 15 features
    
    # ═══════════════════════════════════════════════════════════
    # 5. FEATURES DE CONTEXTE (12)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_context_features(self, state: GameState) -> List[float]:
        """
        Extrait les features de contexte général.
        
        Returns:
            Liste de 12 features:
            - [0]: Nombre de joueurs actifs (normalisé)
            - [1-4]: Street one-hot (preflop/flop/turn/river)
            - [5]: Ratio BB/SB
            - [6-11]: Nombre de joueurs (2-6+, one-hot)
        """
        features = []
        
        # Nombre de joueurs actifs (1)
        num_players = state.num_active_players
        features.append(num_players / 9.0)  # Normalisation (max 9 joueurs)
        
        # Street one-hot (4)
        for street in self.STREETS:
            features.append(1.0 if state.street == street else 0.0)
        
        # Ratio BB/SB (1)
        bb_sb_ratio = state.big_blind / max(state.small_blind, 1)
        features.append(min(bb_sb_ratio / 3.0, 1.0))  # Normalisation (ratio typique = 2)
        
        # Nombre de joueurs one-hot (6): 2, 3, 4, 5, 6, 6+
        for n in range(2, 7):
            features.append(1.0 if num_players == n else 0.0)
        features.append(1.0 if num_players > 6 else 0.0)
        
        return features  # 12 features
    
    # ═══════════════════════════════════════════════════════════
    # 6. FEATURES DE THÉORIE DU JEU (20)
    # ═══════════════════════════════════════════════════════════
    
    def _extract_game_theory_features(self, state: GameState) -> List[float]:
        """
        Extrait les features de théorie du jeu avancées.
        
        Returns:
            Liste de 32 features:
            - [0]: EV estimé du call
            - [1]: EV estimé du fold
            - [2]: EV estimé du raise
            - [3]: Fold equity estimée
            - [4]: Implied odds
            - [5]: Reverse implied odds
            - [6]: Commitment level (pot commited?)
            - [7-9]: Range advantage (nutted/medium/weak)
            - [10-12]: Board coverage (high/medium/low)
            - [13-15]: Polarisation (polarized/merged/capped)
            - [16-19]: Betting patterns (value/bluff/balanced/exploitative)
            - [20]: MDF (Minimum Defense Frequency)
            - [21]: Alpha (Bluff Breakeven Threshold)
            - [22]: Bet-to-Pot Ratio (normalisé)
            - [23]: Pot Geometry (sizing optimal multi-street)
            - [24]: Equity Realization
            - [25]: Blocker Effects
            - [26]: Protection Need
            - [27]: Nut Advantage
            - [28]: Leverage
            - [29]: Effective Stack Depth Ratio (SPR)
            - [30]: Check-Raise Signal
            - [31]: Equity Denial
        """
        features = []
        
        # EV estimations améliorées (3)
        equity = self._estimate_equity(state.hole_cards, state.board, state.num_active_players)
        pot_odds = state.pot_odds
        
        # EV call = equity × pot_total - (1 - equity) × amount_to_call, normalisé [0, 1]
        pot_total = state.pot_size + state.amount_to_call
        ev_call_raw = equity * pot_total - (1 - equity) * state.amount_to_call
        max_val = max(pot_total, state.amount_to_call, 1)
        ev_call = (ev_call_raw / max_val + 1) / 2  # Centré sur 0.5 = breakeven
        ev_call = max(0.0, min(1.0, ev_call))
        features.append(ev_call)
        
        # EV fold = coût du fold (fraction du stack déjà investie dans le pot)
        invested = max(0, state.pot_size - state.stack)
        total_initial = state.stack + invested
        ev_fold = invested / max(total_initial, 1)  # 0 = fold gratuit, ~1 = tout investi
        features.append(ev_fold)
        
        # EV raise = fold_equity × gain_si_fold + (1 - fold_equity) × gain_si_call
        fold_equity = self._estimate_fold_equity(state)
        normalizer = max(state.pot_size + state.stack, 1)
        gain_if_fold = state.pot_size / normalizer
        gain_if_call = equity * (state.pot_size + state.amount_to_call) / normalizer
        ev_raise = fold_equity * gain_if_fold + (1 - fold_equity) * gain_if_call
        ev_raise = max(0.0, min(1.0, ev_raise))
        features.append(ev_raise)
        
        # Fold equity (1)
        features.append(fold_equity)
        
        # Implied odds (1)
        implied_odds = self._calculate_implied_odds(state, equity)
        features.append(implied_odds)
        
        # Reverse implied odds (1)
        reverse_implied_odds = self._calculate_reverse_implied_odds(state, equity)
        features.append(reverse_implied_odds)
        
        # Commitment level (1)
        # Pourcentage du stack déjà investi (clampé à 0 minimum)
        invested = max(0, state.pot_size - state.stack)
        commitment = invested / max(state.stack + invested, 1)
        features.append(min(max(commitment, 0.0), 1.0))
        
        # Range advantage (3)
        range_adv = self._estimate_range_advantage(state)
        features.extend([
            range_adv['nutted'],
            range_adv['medium'],
            range_adv['weak']
        ])
        
        # Board coverage (3)
        board_cov = self._estimate_board_coverage(state)
        features.extend([
            board_cov['high'],
            board_cov['medium'],
            board_cov['low']
        ])
        
        # Polarisation (3)
        polarization = self._estimate_polarization(state)
        features.extend([
            polarization['polarized'],
            polarization['merged'],
            polarization['capped']
        ])
        
        # Betting patterns (4)
        patterns = self._analyze_betting_patterns(state)
        features.extend([
            patterns['value'],
            patterns['bluff'],
            patterns['balanced'],
            patterns['exploitative']
        ])
        
        # ─── GTO Fundamentals (4) ───────────────────────────
        
        # MDF (Minimum Defense Frequency)
        # = 1 - α = 1 - bet/(pot+bet) = pot/(pot+bet)
        # Face à un bet, on DOIT défendre au moins MDF% pour empêcher
        # les bluffs adverses d'être auto-profitables.
        # 0 quand pas de bet à affronter (= pas de contrainte de défense)
        bet_size = state.amount_to_call
        if bet_size > 0:
            mdf = state.pot_size / (state.pot_size + bet_size)
        else:
            mdf = 0.0
        features.append(max(0.0, min(1.0, mdf)))
        
        # Alpha (Bluff Breakeven Threshold)
        # = bet/(pot+bet) = fréquence de fold minimale du vilain
        # pour qu'un bluff soit breakeven. Si fold_equity > α, bluff est +EV.
        # Utilise le dernier sizing agressif (ou amount_to_call si on est le caller)
        effective_bet = state.get_last_aggression_amount()
        if effective_bet <= 0:
            effective_bet = bet_size
        if effective_bet > 0:
            alpha = effective_bet / (state.pot_size + effective_bet)
        else:
            alpha = 0.0
        features.append(max(0.0, min(1.0, alpha)))
        
        # Bet-to-Pot Ratio (normalisé [0, 1] via tanh-like scaling)
        # Sizing relatif au pot : 0 = no bet, 0.5 ≈ pot-size bet, ~1 = overbet
        if effective_bet > 0 and state.pot_size > 0:
            raw_ratio = effective_bet / state.pot_size
            # Sigmoid-like : ratio 1x → 0.5, ratio 2x → 0.67, ratio 3x → 0.75
            bet_to_pot = raw_ratio / (1.0 + raw_ratio)
        else:
            bet_to_pot = 0.0
        features.append(max(0.0, min(1.0, bet_to_pot)))
        
        # Pot Geometry (multi-street optimal sizing)
        # Fraction géométrique = (S+P)/P ^ (1/n) - 1, où n = streets restantes
        # Normalisé en comparant le sizing réel au sizing géométrique optimal.
        # Score > 0.5 = plus gros que géométrique (polarisant)
        # Score < 0.5 = plus petit que géométrique (merged)
        # Score = 0.5 = sizing géométrique parfait
        streets_remaining = {'preflop': 3, 'flop': 2, 'turn': 1, 'river': 0}
        n_streets = streets_remaining.get(state.street, 0)
        
        if n_streets > 0 and state.pot_size > 0 and state.stack > 0:
            # Sizing géométrique optimal pour get all-in à la river
            total_ratio = (state.stack + state.pot_size) / state.pot_size
            geo_fraction = total_ratio ** (1.0 / n_streets) - 1.0
            geo_bet = state.pot_size * geo_fraction  # Bet optimal
            
            if effective_bet > 0 and geo_bet > 0:
                # Ratio sizing_réel / sizing_géométrique, centré sur 0.5
                ratio = effective_bet / geo_bet
                # Sigmoid centré : ratio=1 → 0.5, ratio=2 → 0.67, ratio=0.5 → 0.33
                pot_geometry = ratio / (1.0 + ratio)
            else:
                pot_geometry = 0.5  # Pas de bet = neutre
        else:
            pot_geometry = 0.5  # River ou data manquante = neutre
        features.append(max(0.0, min(1.0, pot_geometry)))
        
        # ─── Tier 1 GTO Advanced (3) ────────────────────────
        
        # Equity Realization
        # Fraction de l'equity brute qu'on réalise réellement.
        # Position IP réalise ~100%, OOP ~65-80%. Les draws réalisent moins
        # (car ils ne font pas toujours leur tirage). Les mains nutted réalisent plus.
        # Formule : base_realization × position_mult × hand_type_mult
        
        is_ip = 1.0 if state.position in ['BTN', 'CO'] else 0.0
        
        # Base : IP = 0.95, OOP = 0.70
        eq_real_base = 0.95 if is_ip else 0.70
        
        if len(state.board) >= 3:
            flush_made, flush_draw = self._check_flush(state.hole_cards, state.board)
            straight_made, straight_draw = self._check_straight(state.hole_cards, state.board)
            hand_strength = self._evaluate_hand_strength(state.hole_cards, state.board)
            
            # Les draws réalisent moins (need to hit), sauf si gros draws
            if flush_draw > 0 or straight_draw > 0:
                draw_penalty = 0.85  # Les draws réalisent ~85% de leur equity
                if flush_draw > 0 and straight_draw > 0:
                    draw_penalty = 0.92  # Combo draw réalise mieux
            elif hand_strength > 0.8:
                draw_penalty = 1.05  # Mains très fortes = surréalisation
            elif hand_strength < 0.3:
                draw_penalty = 0.75  # Mains faibles = sous-réalisation
            else:
                draw_penalty = 1.0  # Mains moyennes = neutre
        else:
            draw_penalty = 1.0  # Preflop = neutre
        
        # Multi-way penalty : plus d'adversaires = moins de réalisation
        mw_mult = 1.0 - 0.05 * max(0, state.num_active_players - 2)
        
        equity_realization = eq_real_base * draw_penalty * max(mw_mult, 0.7)
        features.append(max(0.0, min(1.0, equity_realization)))
        
        # Blocker Effects
        # Est-ce que nos cartes bloquent les mains fortes du vilain ?
        # Exemples : avoir A♠ quand le board a 3 spades → bloque la nut flush
        #            avoir un K sur un board K-high → bloque top pair
        # Score : 0 = aucun blocker, 1 = blockers très puissants
        blocker_score = 0.0
        
        if len(state.board) >= 3:
            board_suits = [self._get_suit_char(c) for c in state.board]
            hero_suits = [self._get_suit_char(c) for c in state.hole_cards]
            hero_ranks = [self._card_rank_to_value(c) for c in state.hole_cards]
            board_ranks = [self._card_rank_to_value(c) for c in state.board]
            
            # 1. Nut flush blocker : on a l'As de la couleur dominante du board
            suit_counts = {}
            for s in board_suits:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            dominant_suit = max(suit_counts, key=suit_counts.get)
            dominant_count = suit_counts[dominant_suit]
            
            if dominant_count >= 3:  # Board monotone ou near-monotone
                # On a l'As de cette couleur → nut flush blocker (+0.4)
                for i, hs in enumerate(hero_suits):
                    if hs == dominant_suit and hero_ranks[i] == 14:
                        blocker_score += 0.4
                    elif hs == dominant_suit and hero_ranks[i] >= 12:
                        blocker_score += 0.2  # K/Q de la couleur
            elif dominant_count >= 2:  # Flush draw possible
                for i, hs in enumerate(hero_suits):
                    if hs == dominant_suit and hero_ranks[i] == 14:
                        blocker_score += 0.2
            
            # 2. Top card blocker : on a la même carte que le board high card
            max_board = max(board_ranks) if board_ranks else 0
            for hr in hero_ranks:
                if hr == max_board and hr >= 12:  # On bloque top pair si on a K/A du board
                    blocker_score += 0.2
            
            # 3. Set blocker : on a une carte qui empêche un set
            for hr in hero_ranks:
                if hr in board_ranks and hr >= 10:
                    blocker_score += 0.1
            
            # 4. Straight blocker : on a des cartes qui se connectent au board
            board_sorted = sorted(board_ranks)
            for hr in hero_ranks:
                # Cartes entre les cartes du board (bloquent les suites)
                if len(board_sorted) >= 2:
                    for j in range(len(board_sorted) - 1):
                        gap = board_sorted[j+1] - board_sorted[j]
                        if 1 < gap <= 3 and board_sorted[j] < hr < board_sorted[j+1]:
                            blocker_score += 0.1
        
        features.append(max(0.0, min(1.0, blocker_score)))
        
        # Protection Need
        # Est-ce que notre main a besoin de protection (i.e. doit-on bet pour deny equity) ?
        # Top pair sur wet board = gros besoin de protection
        # Nuts ou très faible = pas de besoin (nuts n'a pas peur, air n'a rien à protéger)
        # Score : 0 = pas de besoin, 1 = besoin urgent de protéger
        protection = 0.0
        
        if len(state.board) >= 3:
            hand_strength = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex = self._analyze_board_texture(state.board)
            
            # Zone de vulnérabilité : mains moyennes-fortes (0.4 - 0.85)
            # Les nuts (>0.85) n'ont pas besoin de protection
            # Les airs (<0.4) n'ont rien à protéger
            if 0.4 <= hand_strength <= 0.85:
                # Base proportionnelle à la force (plus c'est fort, plus on veut protéger)
                # Pic de protection autour de 0.6-0.7 (top pair / overpair)
                protection = 1.0 - abs(hand_strength - 0.65) / 0.25
                protection = max(0.0, protection)
                
                # Amplification par la dangerosité du board
                # Board wet/coordonné = plus de draws adverses = plus besoin de protéger
                board_danger = board_tex.get('wet', 0) * 0.4 + board_tex.get('coordinated', 0) * 0.3
                protection *= (1.0 + board_danger)
                
                # Streets restantes : plus il reste de streets, plus on doit protéger
                street_mult = {'flop': 1.0, 'turn': 0.7, 'river': 0.0}
                protection *= street_mult.get(state.street, 0.5)
            
            elif hand_strength > 0.85:
                protection = 0.1  # Nuts = très peu de besoin
            else:
                protection = 0.0  # Air = pas de besoin
        
        features.append(max(0.0, min(1.0, protection)))
        
        # ─── Tier 2 GTO Advanced (3) ────────────────────────
        
        # Nut Advantage
        # Est-ce que notre range contient plus de mains nutted que le vilain ?
        # Basé sur : position + board texture + board height
        # IP sur Ace-high board en SRP = gros nut advantage (on a plus d'Ax)
        # OOP sur board connecté = moins de nut advantage
        nut_adv = 0.5  # Neutre par défaut
        
        if len(state.board) >= 3:
            board_ranks_na = [self._card_rank_to_value(c) for c in state.board]
            max_board_rank = max(board_ranks_na)
            board_tex_na = self._analyze_board_texture(state.board)
            
            # A-high / K-high boards favorisent l'IP (3-bettor range)
            if max_board_rank >= 14:  # Ace-high
                nut_adv = 0.70 if state.position in ['BTN', 'CO'] else 0.55
            elif max_board_rank >= 13:  # King-high
                nut_adv = 0.65 if state.position in ['BTN', 'CO'] else 0.50
            elif max_board_rank <= 8:  # Low board — favorise le caller (BB/SB)
                nut_adv = 0.35 if state.position in ['BTN', 'CO'] else 0.60
            else:
                nut_adv = 0.50  # Mid board = neutre
            
            # Board connecté réduit le nut advantage (plus de combos possibles)
            coord = board_tex_na.get('coordinated', 0)
            nut_adv -= coord * 0.15
            
            # Board monotone réduit le nut advantage (flush possible)
            suit_counts_na = {}
            for c in state.board:
                s = self._get_suit_char(c)
                suit_counts_na[s] = suit_counts_na.get(s, 0) + 1
            if max(suit_counts_na.values()) >= 3:
                nut_adv -= 0.1
        
        features.append(max(0.0, min(1.0, nut_adv)))
        
        # Leverage
        # Capacité à mettre de la pression sur les streets futures.
        # Leverage = f(stack_restant, streets_restantes)
        # Plus il reste de stack ET de streets, plus on a de leverage (bluffs crédibles).
        # Formule : stack_to_pot_ratio normalisé × streets_factor
        streets_remaining_lev = {'preflop': 3, 'flop': 2, 'turn': 1, 'river': 0}
        n_streets_lev = streets_remaining_lev.get(state.street, 0)
        
        if state.pot_size > 0 and n_streets_lev > 0:
            spr = state.stack / state.pot_size  # Stack-to-Pot Ratio
            # SPR de 10+ = beaucoup de leverage, SPR de 1 = presque aucun
            spr_norm = min(spr / 10.0, 1.0)  # Normalisé : 10 SPR → 1.0
            streets_factor = n_streets_lev / 3.0  # Preflop=1.0, flop=0.67, turn=0.33
            leverage = spr_norm * streets_factor
        else:
            leverage = 0.0  # River ou pas de pot = pas de leverage
        
        features.append(max(0.0, min(1.0, leverage)))
        
        # Effective Stack Depth Ratio (SPR normalisé)
        # SPR = Stack / Pot. Indicateur clé pour toute décision post-flop.
        # SPR < 2 = commit zone (top pair suffit), SPR > 10 = deep (set mining)
        # Normalisé avec sigmoid : SPR 4 → 0.50, SPR 1 → 0.20, SPR 10 → 0.77
        if state.pot_size > 0:
            raw_spr = state.stack / state.pot_size
            eff_spr = raw_spr / (raw_spr + 4.0)  # Sigmoid centrée sur SPR=4
        else:
            eff_spr = 1.0  # Pas de pot = stack immense relativement
        
        features.append(max(0.0, min(1.0, eff_spr)))
        
        # ─── Tier 3 GTO (2) ──────────────────────────────────
        
        # Check-Raise Signal
        # Détecte si c'est un bon spot pour check-raise basé sur :
        # 1. Position OOP (condition nécessaire pour check-raise)
        # 2. Main forte mais non-nuts (veut build le pot)
        # 3. Board texture favorable (dry = mieux pour check-raise)
        cr_signal = 0.0
        
        is_oop = state.position in ['SB', 'BB', 'UTG']
        
        if is_oop and len(state.board) >= 3:
            hand_strength_cr = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex_cr = self._analyze_board_texture(state.board)
            
            # Zone idéale pour check-raise : mains fortes (0.65-0.90)
            # ou semi-bluffs avec draws
            if 0.65 <= hand_strength_cr <= 0.90:
                cr_signal = 0.7
                # Board dry = meilleur pour check-raise (moins de scare cards)
                wet = board_tex_cr.get('wet', 0)
                cr_signal *= (1.0 - wet * 0.3)
            
            # Semi-bluff check-raise avec draws
            elif hand_strength_cr < 0.4 and len(state.board) >= 3:
                flush_m, flush_d = self._check_flush(state.hole_cards, state.board)
                straight_m, straight_d = self._check_straight(state.hole_cards, state.board)
                if flush_d > 0 or straight_d > 0:
                    cr_signal = 0.5  # Semi-bluff check-raise viable
                    if flush_d > 0 and straight_d > 0:
                        cr_signal = 0.65  # Combo draw = excellent check-raise
            
            # Pas de bet à affronter = on ne peut pas check-raise
            if state.amount_to_call <= 0:
                cr_signal *= 0.3  # Réduire si personne n'a bet
        
        features.append(max(0.0, min(1.0, cr_signal)))
        
        # Equity Denial
        # Gain estimé en faisant folder les draws adverses.
        # Plus il y a de draws possibles ET que notre main est vulnérable,
        # plus la valeur de deny l'equity est élevée.
        # Score = 0 : aucun intérêt à deny equity (nuts ou air)
        # Score = 1 : très profitable de faire folder les draws
        eq_denial = 0.0
        
        if len(state.board) >= 3 and state.street != 'river':  # Pas de denial à la river
            hand_strength_ed = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex_ed = self._analyze_board_texture(state.board)
            
            # Seules les mains moyennes-fortes bénéficient de l'equity denial
            # (air n'a rien à protéger, nuts veut du value)
            if 0.45 <= hand_strength_ed <= 0.85:
                # Base : proportionnelle à la force de main
                base_denial = hand_strength_ed * 0.8
                
                # Amplification par la dangerosité du board
                # Board wet = beaucoup de draws adverses = forte valeur de denial
                wet_ed = board_tex_ed.get('wet', 0)
                coord_ed = board_tex_ed.get('coordinated', 0)
                draw_density = wet_ed * 0.5 + coord_ed * 0.3
                
                # Flush draws on board = plus de denial value
                suit_counts_ed = {}
                for c in state.board:
                    s = self._get_suit_char(c)
                    suit_counts_ed[s] = suit_counts_ed.get(s, 0) + 1
                if 2 in suit_counts_ed.values():
                    draw_density += 0.15  # Flush draw possible
                
                eq_denial = base_denial * (1.0 + draw_density)
                
                # Nombre d'adversaires : plus ils sont nombreux, plus la denial est importante
                eq_denial *= (1.0 + 0.1 * max(0, state.num_active_players - 2))
        
        features.append(max(0.0, min(1.0, eq_denial)))
        
        return features  # 32 features
    
    # ═══════════════════════════════════════════════════════════
    # MÉTHODES UTILITAIRES
    # ═══════════════════════════════════════════════════════════
    
    def _get_rank_char(self, card: str) -> str:
        """
        Détecte intelligemment où est le rang (Rank) dans la string.
        Gère 'Tc' (Pluribus) et 'CT' (RLCard).
        """
        if not card: return '2' # Fallback
        
        # Les caractères qui sont forcément des rangs
        valid_ranks = set('23456789TJQKA')
        
        # Si le premier caractère est un rang (Format Pluribus 'Tc')
        if card[0].upper() in valid_ranks:
            return card[0].upper()
            
        # Sinon, on suppose que c'est le deuxième (Format RLCard 'CT')
        if len(card) > 1 and card[1].upper() in valid_ranks:
            return card[1].upper()
            
        return '2' # Par défaut si parsing échoue

    def _get_suit_char(self, card: str) -> str:
        """
        Détecte intelligemment la couleur.
        """
        if not card: return 's'
        
        valid_ranks = set('23456789TJQKA')
        
        # Si le premier est un rang, la couleur est le 2ème (Pluribus)
        if card[0].upper() in valid_ranks:
            return card[1].lower() if len(card) > 1 else 's'
        
        # Sinon la couleur est le premier (RLCard)
        return card[0].lower()
    
    def _card_rank_to_value(self, card: str) -> int:
        rank_char = self._get_rank_char(card) 
        
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(rank_char, 0)
    
    def _get_preflop_hand_string(self, hole_cards: List[str]) -> str:
        """Convertit les hole cards en string de main preflop (ex: 'AKs')."""
        if len(hole_cards) < 2:
            return ''
        
        r1_char = self._get_rank_char(hole_cards[0])
        r2_char = self._get_rank_char(hole_cards[1])
        
        s1 = self._get_suit_char(hole_cards[0])
        s2 = self._get_suit_char(hole_cards[1])
        
        suited_str = 's' if s1 == s2 else 'o'
        
        val1 = self._card_rank_to_value(hole_cards[0])
        val2 = self._card_rank_to_value(hole_cards[1])
        
        if val1 >= val2:
            return f"{r1_char}{r2_char}{suited_str}"
        else:
            return f"{r2_char}{r1_char}{suited_str}"
    
    
    def _evaluate_hand_strength(self, hole_cards: List[str], board: List[str]) -> float:
        """
        Évalue la force d'une main avec treys.
        
        Args:
            hole_cards: Cartes privées ['C5', 'Sa'] ou ['5c', 'As']
            board: Cartes communes ['D6', 'H9', 'C8'] ou ['6d', '9h', '8c']
        
        Returns:
            Score normalisé [0, 1] (1 = nuts)
        """
        if len(board) < 3:
            return 0.0  # Pas de board = pas d'évaluation
        
        def normalize_card(card: str) -> str:
            """
            Normalise une carte au format treys : 'As', 'Kh', 'Qd', 'Jc'.
            
            Formats acceptés :
            - 'As', 'Kh' (standard)
            - 'SA', 'HK' (RLCard inversé majuscule)
            - 'Sa', 'Hk' (RLCard inversé minuscule)
            - 'C5', 'D6' (RLCard inversé)
            
            Returns:
                Carte normalisée : 'As', '5c', etc.
            """
            if not card or len(card) != 2:
                raise ValueError(f"Carte invalide: '{card}'")
            
            # Détection du format
            char1, char2 = card[0], card[1]
            
            # Format standard : '5c', 'As', 'Kh'
            # Premier caractère = rang (A,K,Q,J,T,2-9)
            if char1.upper() in 'AKQJT23456789':
                rank = char1.upper()
                suit = char2.lower()
            
            # Format inversé : 'C5', 'Sa', 'Hk'
            # Premier caractère = couleur (S,H,D,C)
            elif char1.upper() in 'SHDC':
                suit = char1.lower()
                rank = char2.upper()
            
            else:
                raise ValueError(f"Format de carte inconnu: '{card}'")
            
            return rank + suit
        
        try:
            # Normalisation des cartes
            hand_norm = [normalize_card(c) for c in hole_cards]
            board_norm = [normalize_card(c) for c in board]
            
            # Conversion en entiers treys
            hand_ints = [Card.new(c) for c in hand_norm]
            board_ints = [Card.new(c) for c in board_norm]
            
            score = self.evaluator.evaluate(board_ints, hand_ints)
            
            # Normalisation : score treys ∈ [1, 7462]
            # 1 = Royal Flush, 7462 = 7-high
            normalized = 1.0 - (score - 1) / 7461
            return max(0.0, min(1.0, normalized))
        
        except Exception as e:
            print(f"⚠️  Erreur évaluation main: {e}")
            print(f"   hole_cards={hole_cards}, board={board}")
            return 0.5  # Fallback neutre

    def _estimate_equity(self, hole_cards: List[str], board: List[str], num_opponents: int) -> float:
        """
        Estime l'equity (0-1) en combinant Force Actuelle + Potentiel (Tirages).
        """
        if len(hole_cards) < 2:
            return 0.0
        
        # --- CAS 1 : PREFLOP (Pas de changement majeur) ---
        if not board:
            hand_str = self._get_preflop_hand_string(hole_cards)
            if hand_str in self.PREMIUM_HANDS: base = 0.80
            elif hand_str in self.STRONG_HANDS: base = 0.60
            elif hand_str.endswith('s'): base = 0.50 # Bonus pour suited
            elif 'A' in hand_str or 'K' in hand_str: base = 0.45
            else: base = 0.35
            
            # Ajustement nb joueurs (plus on est nombreux, moins on a d'equity brute)
            return base * (0.95 ** max(0, num_opponents - 1))

        # --- CAS 2 : POSTFLOP (La grosse correction) ---
        
        # A. La force actuelle (calculée avec Treys ou la méthode corrigée)
        current_strength = self._evaluate_hand_strength(hole_cards, board)
        
        # B. Le potentiel (Draws)
        draw_bonus = 0.0
        
        # On ne calcule les bonus que si la main n'est pas déjà "Faite" (Brelan ou mieux)
        if current_strength < 0.7:
            # Check Flush Draw
            flush_made, flush_draw = self._check_flush(hole_cards, board)
            if flush_draw > 0: 
                draw_bonus += 0.20  # ~20% d'equity pour un tirage couleur
            
            # Check Straight Draw
            straight_made, straight_draw = self._check_straight(hole_cards, board)
            if straight_draw > 0:
                draw_bonus += 0.10  # ~10% pour un tirage quinte
                
            # Check Overcards (Si j'ai AK sur un board 2-5-9)
            overcards = self._count_overcards(hole_cards, board)
            if overcards > 0:
                draw_bonus += 0.03 * overcards # Petit bonus

        # C. Total Brut
        raw_equity = current_strength + draw_bonus
        
        # D. Plafond (On ne peut pas dépasser 99%)
        raw_equity = min(raw_equity, 0.99)
        
        # E. Ajustement final selon le nombre d'adversaires
        # L'equity se dilue quand il y a beaucoup de monde
        final_equity = raw_equity * (0.90 ** max(0, num_opponents - 1))
        
        return final_equity
    
    def _check_flush(self, hole_cards: List[str], board: List[str]) -> Tuple[float, float]:
        """Vérifie flush made et flush draw."""
        if len(board) < 3:
            return 0.0, 0.0
        
        all_cards = hole_cards + board
        suits = [self._get_suit_char(c) for c in all_cards]
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        max_suit_count = max(suit_counts.values())
        
        flush_made = 1.0 if max_suit_count >= 5 else 0.0
        flush_draw = 1.0 if max_suit_count == 4 else 0.0
        
        return flush_made, flush_draw
    
    def _check_straight(self, hole_cards: List[str], board: List[str]) -> Tuple[float, float]:
        """
        Vérifie straight made et straight draw avec précision professionnelle.
        
        Améliorations V3 :
        - Dégrade les suites A-2-3-4 et J-Q-K-A en Gutshot (4 outs seulement).
        - Promeut les Double Gutshots en OESD (8 outs).
        
        Returns:
            straight_made (0.0 ou 1.0)
            straight_draw (1.0 = 8 outs+, 0.5 = 4 outs, 0.0 = Rien)
        """
        if len(board) < 3:
            return 0.0, 0.0
        
        all_cards = hole_cards + board
        values = set([self._card_rank_to_value(c) for c in all_cards])
        
        # Gestion de l'As (14 et 1)
        if 14 in values:
            values.add(1)
            
        ranks = sorted(list(values))
        
        straight_made = 0.0
        
        # 1. DÉTECTION SUITE FAITE
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                straight_made = 1.0
                break 
        
        # Si suite faite, on s'arrête là (le draw vaut 0 car on a déjà mieux)
        if straight_made == 1.0:
            return 1.0, 0.0

        # 2. DÉTECTION TIRAGE PRÉCISE
        gutshot_count = 0
        is_oesd = False
        
        # On regarde les fenêtres de 4 cartes
        for i in range(len(ranks) - 3):
            window = ranks[i:i+4]
            gap = window[-1] - window[0]
            
            # Cas A : 4 cartes qui se suivent (ex: 4,5,6,7)
            if gap == 3:
                # PIÈGE : Si la suite est collée au bord (A-2-3-4 ou J-Q-K-A)
                # Ce n'est pas un OESD (8 outs), c'est un One-Ended (4 outs)
                if 1 in window or 14 in window:
                    gutshot_count += 1
                else:
                    is_oesd = True # C'est un vrai 4-5-6-7 au milieu du paquet
            
            # Cas B : Trou de 1 carte (ex: 4,5,7,8) -> Gutshot
            elif gap == 4:
                gutshot_count += 1
        
        # 3. SCORE FINAL
        # Un OESD vaut 1.0
        # UN Double Gutshot (2 gutshots cumulés) vaut aussi 1.0 (car 8 outs)
        if is_oesd or gutshot_count >= 2:
            straight_draw = 1.0
        elif gutshot_count > 0:
            straight_draw = 0.5
        else:
            straight_draw = 0.0
            
        return straight_made, straight_draw
    
    def _count_overcards(self, hole_cards: List[str], board: List[str]) -> int:
        """Compte les overcards au board."""
        if not board:
            return 0
        
        board_max = max([self._card_rank_to_value(c) for c in board])
        overcards = sum(1 for c in hole_cards if self._card_rank_to_value(c) > board_max)
        
        return overcards
    
    def _check_hand_made(self, hole_cards: List[str], board: List[str]) -> Tuple[float, float, float, float]:
        """Vérifie pair, two_pair, trips, quads."""
        all_cards = hole_cards + board
        ranks = [self._card_rank_to_value(c) for c in all_cards]
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        
        counts = list(rank_counts.values())
        
        pair = 1.0 if 2 in counts else 0.0
        two_pair = 1.0 if counts.count(2) >= 2 else 0.0
        trips = 1.0 if 3 in counts else 0.0
        quads = 1.0 if 4 in counts else 0.0
        
        return pair, two_pair, trips, quads
    
    def _analyze_board_texture(self, board: List[str]) -> Dict[str, float]:
        """Analyse la texture du board."""
        if len(board) < 3:
            return {
                'coordinated': 0.0,
                'wet': 0.0,
                'paired': 0.0,
                'high_cards': 0.0,
                'monotone': 0.0
            }
        
        ranks = [self._card_rank_to_value(c) for c in board]
        suits = [self._get_suit_char(c) for c in board]
        
        # Coordinated: cartes proches (possibilité de suites)
        ranks_sorted = sorted(ranks)
        max_gap = max([ranks_sorted[i+1] - ranks_sorted[i] for i in range(len(ranks_sorted)-1)])
        coordinated = 1.0 if max_gap <= 2 else 0.0
        
        # Wet: board avec beaucoup de possibilités de draws
        wet = 1.0 if coordinated or len(set(suits)) <= 2 else 0.0
        
        # Paired board
        paired = 1.0 if len(ranks) != len(set(ranks)) else 0.0
        
        # High cards (J+)
        high_cards = sum(1 for r in ranks if r >= 11) / len(board)
        
        # Monotone (toutes même couleur)
        monotone = 1.0 if len(set(suits)) == 1 else 0.0
        
        return {
            'coordinated': coordinated,
            'wet': wet,
            'paired': paired,
            'high_cards': high_cards,
            'monotone': monotone
        }
    
    def _estimate_fold_equity(self, state: GameState) -> float:
        """Estime la fold equity d'une relance (5 facteurs)."""
        # 1. Base inversement proportionnelle à l'aggression déjà montrée
        #    0.6 = ~60% fold frequency HU face à un c-bet standard (GTO baseline)
        #    -0.15 par raise = chaque relance réduit la fold freq de ~15%
        aggression = sum(1 for a in state.actions_this_street 
                         if any(x in a.lower() for x in ['bet', 'raise']))
        base = max(0.0, 0.6 - aggression * 0.15)
        
        # 2. Position (IP = +10% fold equity, adversaire respecte plus)
        position_bonus = 0.1 if state.position in ['BTN', 'CO'] else 0.0
        
        # 3. Board texture (board sec = +10% max, vilain miss plus souvent)
        texture = self._analyze_board_texture(state.board)
        dry_bonus = 0.1 * (1.0 - texture['wet']) if state.board else 0.05
        
        # 4. Multiway penalty (-5% par joueur additionnel au-delà du HU)
        multiway_penalty = 0.05 * max(0, state.num_active_players - 2)
        
        # 5. Sizing : gros bet = plus de pression (+10% par 1x pot, cap +15%)
        last_agg = state.get_last_aggression_amount()
        if last_agg > 0 and state.pot_size > 0:
            sizing_ratio = last_agg / state.pot_size
            sizing_bonus = min(0.15, sizing_ratio * 0.1)
        else:
            sizing_bonus = 0.0
        
        fold_eq = base + position_bonus + dry_bonus + sizing_bonus - multiway_penalty
        return max(0.0, min(1.0, fold_eq))
    
    def _calculate_implied_odds(self, state: GameState, equity: float) -> float:
        """Calcule les implied odds (3 facteurs continus)."""
        # Factor 1: Stack depth (normalisé à 100BB, clampé à 1.0)
        stack_factor = min(state.effective_stack_bb / 100.0, 1.0)
        
        # Factor 2: Draw quality (triangle : pic à equity=0.5, zéro aux extrêmes)
        if equity <= 0.15 or equity >= 0.75:
            draw_factor = 0.0  # Trop faible ou déjà fait
        elif equity < 0.5:
            draw_factor = equity * 2  # 0.15→0.3, 0.5→1.0
        else:
            draw_factor = (1.0 - equity) * 2  # 0.5→1.0, 0.75→0.5
        
        # Factor 3: Streets restantes (flop=max, river=0 par définition)
        street_mult = {'preflop': 0.8, 'flop': 1.0, 'turn': 0.6, 'river': 0.0}
        street_factor = street_mult.get(state.street, 0.5)
        
        implied = stack_factor * draw_factor * street_factor
        return max(0.0, min(1.0, implied))
    
    def _calculate_reverse_implied_odds(self, state: GameState, equity: float) -> float:
        """Calcule les reverse implied odds (gradient continu [0, 1])."""
        texture = self._analyze_board_texture(state.board)
        
        # Mains moyennes = risque maximal de reverse implied odds
        if equity <= 0.3 or equity >= 0.7:
            return 0.0
        
        # Plus l'equity est proche de 0.5, plus le risque est élevé
        strength_risk = 1.0 - abs(equity - 0.5) * 4  # Max à equity=0.5, 0 aux bords
        strength_risk = max(0.0, strength_risk)
        
        # Board dangereux amplifie le risque
        board_danger = (texture['wet'] * 0.4 + texture['coordinated'] * 0.3 
                        + texture['monotone'] * 0.3)
        
        # Plus il reste de streets, plus le risque est élevé
        street_mult = {'preflop': 0.3, 'flop': 1.0, 'turn': 0.7, 'river': 0.2}
        street_factor = street_mult.get(state.street, 0.5)
        
        risk = strength_risk * board_danger * street_factor
        return max(0.0, min(1.0, risk))
    
    def _estimate_range_advantage(self, state: GameState) -> Dict[str, float]:
        """Estime l'avantage de range (board-aware)."""
        in_position = state.position in ['BTN', 'CO']
        facing_aggression = state.amount_to_call > 0
        texture = self._analyze_board_texture(state.board)
        equity = self._estimate_equity(state.hole_cards, state.board, state.num_active_players)
        
        # Base selon position/aggression
        if in_position and not facing_aggression:
            nutted, medium, weak = 0.3, 0.5, 0.2
        elif facing_aggression:
            nutted, medium, weak = 0.4, 0.3, 0.3
        else:
            nutted, medium, weak = 0.2, 0.5, 0.3
        
        # Board de high cards favorise le raiser IP (+15% nutted)
        if texture['high_cards'] > 0.5:
            if in_position:
                nutted += 0.15; weak -= 0.15
            else:
                nutted -= 0.1; weak += 0.1
        
        # Board coordonné favorise le caller OOP (+10% medium)
        if texture['coordinated']:
            if not in_position:
                medium += 0.1; weak -= 0.1
        
        # Equity réelle ajuste la distribution
        if equity > 0.7:
            nutted += 0.2; weak -= 0.2
        elif equity < 0.3:
            nutted -= 0.1; weak += 0.1
        
        # Normalisation (somme = 1)
        total = max(nutted + medium + weak, 0.01)
        return {
            'nutted': max(0.0, min(1.0, nutted / total)),
            'medium': max(0.0, min(1.0, medium / total)),
            'weak': max(0.0, min(1.0, weak / total))
        }
    
    def _estimate_board_coverage(self, state: GameState) -> Dict[str, float]:
        """Estime la couverture du board par notre range (main-aware)."""
        texture = self._analyze_board_texture(state.board)
        equity = self._estimate_equity(state.hole_cards, state.board, state.num_active_players)
        
        # Base selon texture
        if texture['high_cards'] > 0.6:
            high, medium, low = 0.6, 0.3, 0.1
        elif texture['coordinated']:
            high, medium, low = 0.3, 0.5, 0.2
        else:
            high, medium, low = 0.4, 0.4, 0.2
        
        # Equity réelle : notre main couvre-t-elle bien ce board ?
        if equity > 0.65:
            high += 0.2; low -= 0.1; medium -= 0.1
        elif equity > 0.45:
            medium += 0.15; high -= 0.05; low -= 0.1
        elif equity < 0.3:
            low += 0.2; high -= 0.1; medium -= 0.1
        
        # Draws améliorent la couverture medium (+10%)
        flush_m, flush_d = self._check_flush(state.hole_cards, state.board)
        straight_m, straight_d = self._check_straight(state.hole_cards, state.board)
        if flush_d or straight_d:
            medium += 0.1; low -= 0.1
        
        # Normalisation (somme = 1)
        total = max(high + medium + low, 0.01)
        return {
            'high': max(0.0, min(1.0, high / total)),
            'medium': max(0.0, min(1.0, medium / total)),
            'low': max(0.0, min(1.0, low / total))
        }
    
    def _estimate_polarization(self, state: GameState) -> Dict[str, float]:
        """Estime la polarisation de notre range (sizing-aware)."""
        aggression = sum(1 for a in state.actions_this_street if 'raise' in a.lower())
        
        # Sizing analysis : normalisé par 1.5x pot (seuil de polarisation GTO)
        last_agg = state.get_last_aggression_amount()
        if last_agg > 0 and state.pot_size > 0:
            sizing_ratio = last_agg / state.pot_size
            sizing_signal = min(sizing_ratio / 1.5, 1.0)
        else:
            sizing_signal = 0.0
        
        # Base selon aggression count
        if aggression >= 2:
            polarized, merged, capped = 0.6, 0.2, 0.2
        elif aggression == 1:
            polarized, merged, capped = 0.3, 0.5, 0.2
        else:
            polarized, merged, capped = 0.1, 0.4, 0.5
        
        # Sizing shift : gros sizing → +polarisé, -merged
        polarized += sizing_signal * 0.2
        merged -= sizing_signal * 0.15
        capped -= sizing_signal * 0.05
        
        # OOP passif → +capped (check-call = pas de nuts)
        if state.position not in ['BTN', 'CO'] and aggression == 0:
            capped += 0.1; polarized -= 0.1
        
        # Normalisation (somme = 1)
        total = max(polarized + merged + capped, 0.01)
        return {
            'polarized': max(0.0, min(1.0, polarized / total)),
            'merged': max(0.0, min(1.0, merged / total)),
            'capped': max(0.0, min(1.0, capped / total))
        }
    
    def _analyze_betting_patterns(self, state: GameState) -> Dict[str, float]:
        """Analyse les patterns de mise (ratios continus)."""
        actions = state.actions_this_street
        n = max(len(actions), 1)
        
        num_bets = sum(1 for a in actions if 'bet' in a.lower())
        num_raises = sum(1 for a in actions if 'raise' in a.lower())
        num_calls = sum(1 for a in actions if 'call' in a.lower())
        num_checks = sum(1 for a in actions if 'check' in a.lower())
        
        aggressive = num_bets + num_raises
        passive = num_calls + num_checks
        
        # Value : ratio agressif (floor 20%, scale 80%)
        value = aggressive / n * 0.8 + 0.2
        
        # Bluff : overbet signal (floor 20%, gros sizing → +30% max)
        last_agg = state.get_last_aggression_amount()
        if last_agg > 0 and state.pot_size > 0:
            overbet_signal = min(last_agg / state.pot_size, 2.0) / 2.0
            bluff = 0.2 + overbet_signal * 0.3
        else:
            bluff = 0.2 + passive / n * 0.1
        
        # Balanced : mix agressif/passif (floor 10%, scale 40%)
        if n > 1:
            balance_ratio = 1.0 - abs(aggressive - passive) / n
        else:
            balance_ratio = 0.5
        balanced = 0.1 + balance_ratio * 0.4
        
        # Exploitative : pattern unilatéral (floor 10%, scale 50%)
        exploitative = abs(aggressive - passive) / n * 0.5 + 0.1
        
        # Normalisation (somme = 1)
        total = max(value + bluff + balanced + exploitative, 0.01)
        return {
            'value': max(0.0, min(1.0, value / total)),
            'bluff': max(0.0, min(1.0, bluff / total)),
            'balanced': max(0.0, min(1.0, balanced / total)),
            'exploitative': max(0.0, min(1.0, exploitative / total))
        }
    
    def _generate_feature_names(self) -> List[str]:
        """Génère les noms de toutes les features."""
        names = []
        
        # Cartes (22)
        names.extend([
            'card_rank_1', 'card_rank_2', 'suited', 'pocket_pair',
            'preflop_premium', 'preflop_strong', 'postflop_strength', 'equity',
            'flush_made', 'flush_draw', 'straight_made', 'straight_draw',
            'overcards', 'pair', 'two_pair', 'trips', 'quads',
            'board_coordinated', 'board_wet', 'board_paired', 'board_high_cards', 'board_monotone'
        ])
        
        # Position (6)
        names.extend([
            'position_normalized', 'distance_to_button', 'in_position',
            'position_early', 'position_middle', 'position_late'
        ])
        
        # Stack & Pot (12)
        names.extend([
            'stack_bb_normalized', 'pot_bb_normalized', 'spr_normalized', 'pot_odds',
            'amount_to_call_bb_normalized', 'all_in_situation',
            'stack_short', 'stack_medium', 'stack_deep',
            'spr_low', 'spr_medium', 'spr_high'
        ])
        
        # Actions (15)
        names.extend([
            'num_actions_normalized', 'aggression_factor',
            'num_folds', 'num_calls', 'num_raises', 'num_checks', 'num_bets', 'num_allins',
            'last_action_aggressive', 'last_aggression_bb', 'facing_aggression',
            'can_fold', 'can_call', 'can_check', 'can_raise'
        ])
        
        # Contexte (12)
        names.extend([
            'num_active_players',
            'street_preflop', 'street_flop', 'street_turn', 'street_river',
            'bb_sb_ratio',
            'players_2', 'players_3', 'players_4', 'players_5', 'players_6', 'players_6plus'
        ])
        
        # Théorie du jeu (32)
        names.extend([
            'ev_call', 'ev_fold', 'ev_raise', 'fold_equity',
            'implied_odds', 'reverse_implied_odds', 'commitment_level',
            'range_nutted', 'range_medium', 'range_weak',
            'board_coverage_high', 'board_coverage_medium', 'board_coverage_low',
            'polarized', 'merged', 'capped',
            'pattern_value', 'pattern_bluff', 'pattern_balanced', 'pattern_exploitative',
            'mdf', 'alpha', 'bet_to_pot_ratio', 'pot_geometry',
            'equity_realization', 'blocker_effects', 'protection_need',
            'nut_advantage', 'leverage', 'effective_spr',
            'check_raise_signal', 'equity_denial'
        ])
        
        return names
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms de toutes les features."""
        return self.feature_names.copy()









class FeatureExtractor_v2(CardFeatures, SimpleFeature, GameTheoryFeatures):

    PREMIUM_HANDS = {'AA', 'KK', 'QQ', 'JJ', 'AKs'}
    STRONG_HANDS = {'TT', '99', '88', 'AQs', 'AJs', 'KQs', 'AKo', 'AQo'}


    def __init__(self, use_one_hot: bool = True):
        self.evaluator = Evaluator()
        self.use_one_hot = use_one_hot
        self.NUM_FEATURES = 203 if use_one_hot else 99
        
        CardFeatures.__init__(self, self.PREMIUM_HANDS, self.STRONG_HANDS, self.evaluator)
        GameTheoryFeatures.__init__(self, self.PREMIUM_HANDS, self.STRONG_HANDS, self.evaluator)

        self.feature_names = self._generate_feature_names()
        logger.info(f"FeatureExtractor_v2 initialisé avec {self.NUM_FEATURES} features")

    def _extract_one_hot_cards(self, state: GameState, out: np.ndarray, idx: int) -> int:
        """
        Encode les cartes du joueur (52 bits) et du board (52 bits) en One-Hot.
        """
        ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suits = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        
        # Hole cards
        for card in state.hole_cards:
            r = self._get_rank_char(card)
            s = self._get_suit_char(card)
            if r in ranks and s in suits:
                card_idx = suits[s] * 13 + ranks[r]
                out[idx + card_idx] = 1.0
                
        idx += 52
        
        # Board cards
        for card in state.board:
            r = self._get_rank_char(card)
            s = self._get_suit_char(card)
            if r in ranks and s in suits:
                card_idx = suits[s] * 13 + ranks[r]
                out[idx + card_idx] = 1.0
                
        idx += 52
        
        return idx

    def extract(self, state: GameState) -> np.ndarray:
        """
        Extrait toutes les features d'un GameState de façon optimisée.
        
        Args:
            state: GameState standardisé
        
        Returns:
            np.ndarray de shape (99,) avec toutes les features
        """
        # Pré-allocation pure Numpy, ultra rapide
        features_array = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        idx = 0
        
        # 1. Features de cartes (22)
        idx = self._extract_card_features(state, features_array, idx)
        
        # 2. Features de position (6)
        idx = self._extract_position_features(state, features_array, idx)
        
        # 3. Features de stack & pot (12)
        idx = self._extract_stack_pot_features(state, features_array, idx)
        
        # 4. Features d'actions (15)
        idx = self._extract_action_features(state, features_array, idx)
        
        # 5. Features de contexte (12)
        idx = self._extract_context_features(state, features_array, idx)
        
        # 6. Features de théorie du jeu (32)
        idx = self._extract_game_theory_features(state, features_array, idx)
        
        # 7. One-Hot Encoding des Cartes (104)
        if self.use_one_hot:
            idx = self._extract_one_hot_cards(state, features_array, idx)
        
        # Validation super rapide
        if idx != self.NUM_FEATURES:
            logger.error(
                f"Nombre de features incorrect après parsing: {idx} != {self.NUM_FEATURES}"
            )
            raise ValueError(f"Expected {self.NUM_FEATURES} features, got {idx}")
        
        return features_array
    
    def _generate_feature_names(self) -> List[str]:
        """Génère la liste des noms de toutes les features."""
        names = []
        
        # Cartes (22)
        names.extend([
            'card_rank_1', 'card_rank_2',
            'suited', 'pocket_pair',
            'hand_strength_preflop_premium', 'hand_strength_preflop_strong',
            'hand_strength_postflop', 'equity_estimated',
            'flush_draw_made', 'flush_draw_draw',
            'straight_draw_made', 'straight_draw_draw',
            'overcards_to_board',
            'pair_on_board', 'two_pair_on_board', 'trips_on_board', 'quads_on_board',
            'board_coordinated', 'board_wet', 'board_paired', 'board_high_cards', 'board_monotone'
        ])
        
        # Position (6)
        names.extend([
            'position_normalized', 'distance_to_button', 'in_position',
            'position_early', 'position_middle', 'position_late'
        ])
        
        # Stack & Pot (12)
        names.extend([
            'effective_stack_bb', 'pot_size_bb', 'spr',
            'pot_odds', 'amount_to_call_bb', 'all_in_situation',
            'stack_short', 'stack_medium', 'stack_deep',
            'spr_low', 'spr_medium', 'spr_high'
        ])
        
        # Actions (15)
        names.extend([
            'num_actions', 'aggression_factor',
            'num_folds', 'num_calls', 'num_raises', 'num_checks', 'num_bets', 'num_allin',
            'last_action_aggressive', 'last_aggression_amount', 'facing_aggression',
            'can_fold', 'can_check_call', 'can_raise', 'can_allin'
        ])
        
        # Contexte (12)
        names.extend([
            'num_active_players',
            'street_preflop', 'street_flop', 'street_turn', 'street_river',
            'bb_sb_ratio',
            'is_single_raised_pot', 'is_3bet_pot', 'was_multiway_flop', 'players_2', 'players_3_to_4', 'players_5_plus'
        ])
        
        # Théorie du jeu (32)
        names.extend([
            'ev_call', 'ev_fold', 'ev_raise', 'fold_equity',
            'implied_odds', 'reverse_implied_odds', 'commitment_level',
            'range_nutted', 'range_medium', 'range_weak',
            'board_coverage_high', 'board_coverage_medium', 'board_coverage_low',
            'polarized', 'merged', 'capped',
            'pattern_value', 'pattern_bluff', 'pattern_balanced', 'pattern_exploitative',
            'mdf', 'alpha', 'bet_to_pot_ratio', 'pot_geometry',
            'equity_realization', 'blocker_effects', 'protection_need',
            'nut_advantage', 'leverage', 'effective_spr',
            'check_raise_signal', 'equity_denial'
        ])
        
        # One-Hot Cards (104)
        if getattr(self, 'use_one_hot', True):
            ranks_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
            suits_order = ['s', 'h', 'd', 'c']
            
            for s in suits_order:
                for r in ranks_order:
                    names.append(f'hole_{r}{s}')
                    
            for s in suits_order:
                for r in ranks_order:
                    names.append(f'board_{r}{s}')
        
        return names
    
    def get_feature_names(self) -> List[str]:
        """Retourne les noms de toutes les features."""
        return self.feature_names.copy()


# ═══════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 TESTS FeatureExtractor")
    print("=" * 70)
    
    extractor = FeatureExtractor()
    
    # Test 1: Preflop premium hand
    state1 = GameState(
        hole_cards=['As', 'Kd'],
        board=[],
        street='preflop',
        position='BTN',
        num_active_players=3,
        pot_size=150,
        stack=10000,
        big_blind=100,
        small_blind=50,
        amount_to_call=100,
        legal_actions=['fold', 'call', 'raise'],
        actions_this_street=[]
    )
    
    print("\n📊 Test 1: Preflop AKo BTN")
    features1 = extractor.extract(state1)
    print(f"  Shape: {features1.shape}")
    print(f"  Nombre de features: {len(features1)}")
    print(f"  Min/Max: [{features1.min():.3f}, {features1.max():.3f}]")
    print(f"  Premières features: {features1[:10]}")
    assert len(features1) == 99
    print("  ✅ PASS")
    
    # Test 2: Postflop avec flush draw
    state2 = GameState(
        hole_cards=['Qh', 'Jh'],
        board=['Kh', '9c', '4h'],
        street='flop',
        position='CO',
        num_active_players=2, 
        pot_size=500,
        stack=8500,
        big_blind=100,
        small_blind=50,
        amount_to_call=300,
        legal_actions=['fold', 'call', 'raise'],
        actions_this_street=['bet_300']
    )
    
    print("\n📊 Test 2: Flop flush draw")
    features2 = extractor.extract(state2)
    print(f"  Shape: {features2.shape}")
    print(f"  Flush draw detected: {features2[9]}")  # flush_draw feature
    print(f"  Equity: {features2[7]:.3f}")
    print(f"  Pot odds: {features2[25]:.3f}")
    assert features2[9] == 1.0  # Flush draw
    print("  ✅ PASS")
    
    # Test 3: Feature names
    print("\n📊 Test 3: Feature names")
    names = extractor.get_feature_names()
    print(f"  Nombre de noms: {len(names)}")
    print(f"  Premiers noms: {names[:5]}")
    print(f"  Derniers noms: {names[-5:]}")
    assert len(names) == 99
    print("  ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PASSENT !")
    print("=" * 70)
