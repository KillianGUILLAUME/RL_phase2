# features/feature_extractor.py

import numpy as np
from typing import Dict, List, Tuple
from core.game_state import GameState
import logging

from treys import Evaluator, Card

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracteur de features universel pour le poker No-Limit Hold'em.
    
    Compatible avec:
    - Pluribus (via PluribusAdapter)
    - RLCard (via RLCardAdapter)
    
    Features extraites (87 au total):
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
    
    6. THÉORIE DU JEU (20 features)
       - EV estimé
       - Fold equity
       - Implied odds
       - Range advantage
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
    NUM_FEATURES = 87
    
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
            np.ndarray de shape (87,) avec toutes les features
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
            Liste de 20 features:
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
        """
        features = []
        
        # EV estimations simplifiées (3)
        equity = self._estimate_equity(state.hole_cards, state.board, state.num_active_players)
        pot_odds = state.pot_odds
        
        # EV call = equity - pot_odds
        ev_call = max(0, equity - pot_odds)
        features.append(ev_call)
        
        # EV fold = 0 (on perd ce qu'on a déjà mis)
        ev_fold = 0.0
        features.append(ev_fold)
        
        # EV raise = equity * fold_equity_estimation
        fold_equity = self._estimate_fold_equity(state)
        ev_raise = equity * (1 + fold_equity)
        features.append(min(ev_raise, 1.0))
        
        # Fold equity (1)
        features.append(fold_equity)
        
        # Implied odds (1)
        implied_odds = self._calculate_implied_odds(state, equity)
        features.append(implied_odds)
        
        # Reverse implied odds (1)
        reverse_implied_odds = self._calculate_reverse_implied_odds(state, equity)
        features.append(reverse_implied_odds)
        
        # Commitment level (1)
        # Pourcentage du stack déjà investi
        invested = state.pot_size - state.stack
        commitment = invested / max(state.stack + invested, 1)
        features.append(commitment)
        
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
        
        return features  # 20 features
    
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
            
            evaluator = Evaluator()
            score = evaluator.evaluate(board_ints, hand_ints)
            
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
        """Estime la fold equity d'une relance."""
        # Simplifié: basé sur aggression déjà montrée et pot odds offerts
        aggression = sum(1 for a in state.actions_this_street if any(x in a.lower() for x in ['bet', 'raise']))
        
        # Plus il y a eu d'aggression, moins on a de fold equity
        base_fold_equity = 0.5 - (aggression * 0.1)
        
        # Ajustement selon les pot odds (moins on donne de cote, plus on a de fold equity)
        if state.pot_odds < 0.25:
            base_fold_equity += 0.2
        
        return min(max(base_fold_equity, 0.0), 1.0)
    
    def _calculate_implied_odds(self, state: GameState, equity: float) -> float:
        """Calcule les implied odds."""
        # Si on a de l'equity et du stack derrière, on a des implied odds
        if equity > 0.3 and state.effective_stack_bb > 20:
            return min((state.effective_stack_bb - 20) / 100.0, 1.0)
        return 0.0
    
    def _calculate_reverse_implied_odds(self, state: GameState, equity: float) -> float:
        """Calcule les reverse implied odds."""
        # Si on a une main moyenne sur un board dangereux
        texture = self._analyze_board_texture(state.board)
        if 0.3 < equity < 0.7 and (texture['wet'] or texture['coordinated']):
            return 0.5
        return 0.0
    
    def _estimate_range_advantage(self, state: GameState) -> Dict[str, float]:
        """Estime l'avantage de range."""
        # Simplifié: basé sur position et actions
        in_position = state.position in ['BTN', 'CO']
        facing_aggression = state.amount_to_call > 0
        
        if in_position and not facing_aggression:
            return {'nutted': 0.3, 'medium': 0.5, 'weak': 0.2}
        elif facing_aggression:
            return {'nutted': 0.5, 'medium': 0.3, 'weak': 0.2}
        else:
            return {'nutted': 0.2, 'medium': 0.5, 'weak': 0.3}
    
    def _estimate_board_coverage(self, state: GameState) -> Dict[str, float]:
        """Estime la couverture du board par notre range."""
        texture = self._analyze_board_texture(state.board)
        
        if texture['high_cards'] > 0.6:
            return {'high': 0.7, 'medium': 0.2, 'low': 0.1}
        elif texture['coordinated']:
            return {'high': 0.3, 'medium': 0.5, 'low': 0.2}
        else:
            return {'high': 0.4, 'medium': 0.4, 'low': 0.2}
    
    def _estimate_polarization(self, state: GameState) -> Dict[str, float]:
        """Estime la polarisation de notre range."""
        aggression = sum(1 for a in state.actions_this_street if 'raise' in a.lower())
        
        if aggression >= 2:
            return {'polarized': 0.7, 'merged': 0.2, 'capped': 0.1}
        elif aggression == 1:
            return {'polarized': 0.3, 'merged': 0.6, 'capped': 0.1}
        else:
            return {'polarized': 0.1, 'merged': 0.5, 'capped': 0.4}
    
    def _analyze_betting_patterns(self, state: GameState) -> Dict[str, float]:
        """Analyse les patterns de mise."""
        actions = state.actions_this_street
        
        num_bets = sum(1 for a in actions if 'bet' in a.lower())
        num_raises = sum(1 for a in actions if 'raise' in a.lower())
        
        # Patterns simplifiés
        if num_raises >= 2:
            return {'value': 0.6, 'bluff': 0.2, 'balanced': 0.1, 'exploitative': 0.1}
        elif num_bets + num_raises == 1:
            return {'value': 0.4, 'bluff': 0.3, 'balanced': 0.2, 'exploitative': 0.1}
        else:
            return {'value': 0.3, 'bluff': 0.2, 'balanced': 0.3, 'exploitative': 0.2}
    
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
        
        # Théorie du jeu (20)
        names.extend([
            'ev_call', 'ev_fold', 'ev_raise', 'fold_equity',
            'implied_odds', 'reverse_implied_odds', 'commitment_level',
            'range_nutted', 'range_medium', 'range_weak',
            'board_coverage_high', 'board_coverage_medium', 'board_coverage_low',
            'polarized', 'merged', 'capped',
            'pattern_value', 'pattern_bluff', 'pattern_balanced', 'pattern_exploitative'
        ])
        
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
    assert len(features1) == 87
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
    assert len(names) == 87
    print("  ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PASSENT !")
    print("=" * 70)
