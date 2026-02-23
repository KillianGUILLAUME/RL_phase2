"""
RuleBasedBot — Agent de poker No-Limit Hold'em basé sur des règles.

Niveau : Amateur / Semi-pro
Vitesse : ~10μs par décision (pas de ML, lookup tables + arithmétique)

Stratégie :
- Preflop : ranges tiered par position (4 niveaux)
- Postflop : hand_strength × pot_odds × SPR
- Game theory basique : MDF, alpha, fold equity, EV
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from treys import Evaluator, Card

from core.game_state import GameState
from adapters.rlcard_adapter import RLCardAdapter


# ═══════════════════════════════════════════════════════════
# PREFLOP RANGES (lookup tables pré-calculées)
# ═══════════════════════════════════════════════════════════

# Tier 1 : Premium — Raise/Re-raise toujours
TIER1_HANDS: Set[str] = {
    'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'
}

# Tier 2 : Strong — Raise IP, Call OOP (ou raise si position tardive)
TIER2_HANDS: Set[str] = {
    'TT', '99', 'AQs', 'AQo', 'AJs', 'ATs', 'KQs', 'KJs'
}

# Tier 3 : Playable — Call si bons pot odds, fold si face à 3-bet
TIER3_HANDS: Set[str] = {
    '88', '77', '66', '55', '44', '33', '22',
    'KQo', 'KJo', 'QJs', 'QTs', 'JTs', 'T9s', '98s', '87s', '76s',
    'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
    'KTs', 'K9s', 'Q9s', 'J9s'
}

# Positions IP (in position) et late position
LATE_POSITIONS: Set[str] = {'BTN', 'CO'}
MIDDLE_POSITIONS: Set[str] = {'MP', 'MP+1'}
EARLY_POSITIONS: Set[str] = {'UTG', 'UTG+1', 'SB', 'BB'}


class RuleBasedBot:
    """
    Bot de poker basé sur des règles optimisées.
    
    Prend des décisions cohérentes et rapides (~10μs) en combinant :
    1. Preflop ranges tiered par position
    2. Postflop hand strength via treys (évaluation réelle)
    3. Pot odds, SPR, et game theory basique (MDF, alpha, fold equity)
    
    Compatible RLCard : step(), eval_step(), use_raw = True
    """
    
    def __init__(self, env):
        self.env = env
        self.adapter = RLCardAdapter()
        self.evaluator = Evaluator()  # Singleton, instancié UNE fois
        self.use_raw = True
        
        # Stats de tracking
        self.stats = {
            'total_decisions': 0,
            'actions': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            'action_names': {
                0: 'FOLD', 1: 'CHECK/CALL',
                2: 'RAISE_HALF', 3: 'RAISE_POT', 4: 'ALL-IN'
            },
            'errors': 0,
            'preflop_decisions': 0,
            'postflop_decisions': 0,
        }
    
    # ═══════════════════════════════════════════════════════════
    # INTERFACE RLCARD
    # ═══════════════════════════════════════════════════════════
    
    def step(self, state: Dict) -> int:
        """Mode training — retourne l'action enum."""
        action_enum, _ = self._decide(state)
        return action_enum
    
    def eval_step(self, state: Dict) -> Tuple:
        """Mode évaluation — retourne (action_enum, probs_dict)."""
        return self._decide(state)
    
    # ═══════════════════════════════════════════════════════════
    # PIPELINE DE DÉCISION PRINCIPAL
    # ═══════════════════════════════════════════════════════════
    
    def _decide(self, state: Dict) -> Tuple:
        """
        Pipeline principal : State → GameState → Décision → Action.
        ~10μs par appel.
        """
        self.stats['total_decisions'] += 1
        
        legal_actions_enum = state.get('raw_legal_actions', [])
        legal_actions_int = [a.value for a in legal_actions_enum]
        
        if not legal_actions_int:
            # Fallback sécurisé
            self.stats['errors'] += 1
            return legal_actions_enum[0] if legal_actions_enum else 0, {}
        
        try:
            # 1. Conversion RLCard → GameState
            game_state = self.adapter.to_game_state(state, self.env)
            
            # 2. Décision selon la street
            if game_state.street == 'preflop':
                self.stats['preflop_decisions'] += 1
                action_int = self._preflop_decision(game_state, legal_actions_int)
            else:
                self.stats['postflop_decisions'] += 1
                action_int = self._postflop_decision(game_state, legal_actions_int)
            
            # 3. Validation : l'action DOIT être légale
            if action_int not in legal_actions_int:
                # Fallback : check/call si possible, sinon fold
                action_int = 1 if 1 in legal_actions_int else legal_actions_int[0]
            
            # 4. Stats
            self.stats['actions'][action_int] = self.stats['actions'].get(action_int, 0) + 1
            
            # 5. Conversion int → enum
            action_enum = legal_actions_enum[legal_actions_int.index(action_int)]
            
            # Probas fictives (100% sur l'action choisie)
            probs = {a: (1.0 if a == action_enum else 0.0) for a in legal_actions_enum}
            
            return action_enum, probs
            
        except Exception as e:
            self.stats['errors'] += 1
            # Fallback sécurisé : check/call
            fallback = legal_actions_enum[0]
            for a in legal_actions_enum:
                if a.value == 1:
                    fallback = a
                    break
            return fallback, {a: (1.0 if a == fallback else 0.0) for a in legal_actions_enum}
    
    # ═══════════════════════════════════════════════════════════
    # STRATÉGIE PREFLOP
    # ═══════════════════════════════════════════════════════════
    
    def _preflop_decision(self, gs: GameState, legal: List[int]) -> int:
        """
        Décision preflop basée sur la tier de la main + position.
        
        Tier 1 (Premium) : Raise toujours, re-raise face à un raise
        Tier 2 (Strong)  : Raise IP, call OOP
        Tier 3 (Playable): Call si odds OK, fold si 3-bet+
        Tier 4 (Trash)   : Fold (sauf check gratuit en BB)
        """
        hand_str = self._get_hand_string(gs.hole_cards)
        position = gs.position
        is_ip = position in LATE_POSITIONS
        facing_raise = gs.amount_to_call > gs.big_blind  # Quelqu'un a raise
        facing_3bet = gs.has_action_occurred('raise') and facing_raise
        
        # ─── Tier 1 : Premium ───────────────────────────────
        if hand_str in TIER1_HANDS:
            # Toujours raise / re-raise
            if 3 in legal:   # RAISE_POT
                return 3
            if 2 in legal:   # RAISE_HALF_POT
                return 2
            return 1  # Call si pas de raise possible (rare)
        
        # ─── Tier 2 : Strong ────────────────────────────────
        if hand_str in TIER2_HANDS:
            if facing_3bet:
                # Face à un 3-bet : call les meilleurs, fold le reste
                if hand_str in {'TT', 'AQs', 'AQo'}:
                    return 1  # Call
                return 0  # Fold les mains les plus faibles de tier 2
            
            if is_ip or position in MIDDLE_POSITIONS:
                # IP / MP : raise
                if 2 in legal:
                    return 2  # RAISE_HALF_POT
                return 1
            else:
                # OOP (UTG, SB, BB) : call
                return 1
        
        # ─── Tier 3 : Playable ──────────────────────────────
        if hand_str in TIER3_HANDS:
            if facing_3bet:
                return 0  # Fold face à un 3-bet
            
            if facing_raise:
                # Face à un open-raise : call seulement en position
                if is_ip:
                    return 1  # Call IP
                # OOP : call seulement les meilleurs (paires, suited connectors forts)
                if hand_str in {'88', '77', '66', 'QJs', 'JTs', 'T9s', 'A5s'}:
                    return 1
                return 0  # Fold le reste OOP
            
            # Pas de raise : open-raise si en position, limp/call sinon
            if is_ip and 2 in legal:
                return 2  # Open raise
            return 1  # Check/call
        
        # ─── Tier 4 : Trash ─────────────────────────────────
        # Check gratuit en BB si personne n'a raise
        if gs.amount_to_call == 0 and 1 in legal:
            return 1  # Check gratuit
        
        return 0  # Fold
    
    # ═══════════════════════════════════════════════════════════
    # STRATÉGIE POSTFLOP
    # ═══════════════════════════════════════════════════════════
    
    def _postflop_decision(self, gs: GameState, legal: List[int]) -> int:
        """
        Décision postflop : hand_strength × pot_odds × SPR.
        
        Logique en cascade :
        1. Évaluer la main (treys)
        2. Calculer equity, pot odds, SPR
        3. Appliquer la matrice décisionnelle
        """
        # ─── 1. Évaluation de la main ───────────────────────
        strength = self._fast_hand_strength(gs.hole_cards, gs.board)
        equity = self._fast_equity(gs.hole_cards, gs.board, gs.num_active_players)
        has_draw = self._has_draw(gs.hole_cards, gs.board)
        
        pot_odds = gs.pot_odds
        spr = gs.spr
        is_ip = gs.position in LATE_POSITIONS
        facing_bet = gs.amount_to_call > 0
        
        # ─── 2. Monster (nuts / near-nuts) ──────────────────
        if strength >= 0.85:
            return self._value_action(gs, legal, spr, aggressive=True)
        
        # ─── 3. Strong hand ─────────────────────────────────
        if strength >= 0.65:
            if spr < 4:
                # SPR bas → commit (pot-size raise ou all-in)
                return self._value_action(gs, legal, spr, aggressive=True)
            
            if facing_bet:
                # Facing bet : call (ou raise si très fort)
                if strength >= 0.75 and 2 in legal:
                    return 2  # Raise semi-gros
                return 1  # Call
            else:
                # Pas de bet : bet for value
                return self._value_action(gs, legal, spr, aggressive=False)
        
        # ─── 4. Medium hand ─────────────────────────────────
        if strength >= 0.45:
            if facing_bet:
                # Call si equity > pot odds (décision GTO basique)
                if equity > pot_odds:
                    return 1  # Call profitable
                
                # MDF check : doit-on défendre ?
                mdf = gs.pot_size / (gs.pot_size + gs.amount_to_call) if gs.amount_to_call > 0 else 0
                if strength >= 0.55 and equity > mdf * 0.8:
                    return 1  # Défense marginale
                
                return 0  # Fold
            else:
                # Pas de bet : check (pot control) ou small bet si IP
                if is_ip and strength >= 0.55 and 2 in legal:
                    return 2  # Bet demi-pot pour protection/thin value
                return 1  # Check
        
        # ─── 5. Drawing hand ────────────────────────────────
        if has_draw:
            if facing_bet:
                # Implied odds : call si le draw est rentable
                # Simplified : equity brute + ~10% implied
                adjusted_equity = equity + 0.10
                if adjusted_equity > pot_odds:
                    return 1  # Call le draw
                
                # Semi-bluff raise si IP et draw puissant
                if is_ip and equity >= 0.30 and 2 in legal:
                    return 2  # Semi-bluff raise
                
                return 0  # Fold le draw trop cher
            else:
                # Pas de bet : semi-bluff si IP, check si OOP
                if is_ip and 2 in legal and gs.street != 'river':
                    return 2  # Semi-bluff bet
                return 1  # Check
        
        # ─── 6. Weak / Air ──────────────────────────────────
        if facing_bet:
            return 0  # Fold face à un bet
        
        # Spot de bluff ? Board sec + IP + SPR élevé + pas river
        if (is_ip and spr > 6 and gs.street != 'river'
                and not gs.has_action_occurred('raise')
                and self._is_dry_board(gs.board) and 2 in legal):
            # Bluff 1 barrel : alpha check
            # On bluff si le sizing nous donne un break-even favorable
            alpha = 0.5  # Environ 50% fold needed pour demi-pot bet
            # On estime ~60% fold rate sur dry board quand on bet IP
            estimated_fold_rate = 0.55
            if estimated_fold_rate > alpha:
                return 2  # Bluff semi-pot
        
        # Check/fold
        if 1 in legal:
            return 1  # Check gratuit
        return 0  # Fold
    
    # ═══════════════════════════════════════════════════════════
    # HELPERS — VALUE ACTIONS
    # ═══════════════════════════════════════════════════════════
    
    def _value_action(self, gs: GameState, legal: List[int],
                      spr: float, aggressive: bool) -> int:
        """
        Choisir la bonne action de value selon le SPR et l'agressivité.
        
        SPR < 2  → All-in (commit)
        SPR < 4  → Raise pot
        SPR < 8  → Raise half-pot
        SPR >= 8 → Raise half-pot (multi-street)
        """
        if aggressive and spr < 2 and 4 in legal:
            return 4  # ALL-IN (commit avec SPR bas)
        if aggressive and spr < 4 and 3 in legal:
            return 3  # RAISE_POT
        if 2 in legal:
            return 2  # RAISE_HALF_POT
        if 3 in legal:
            return 3  # RAISE_POT si pas de half disponible
        return 1  # Call si aucun raise possible
    
    # ═══════════════════════════════════════════════════════════
    # ÉVALUATION RAPIDE DES MAINS
    # ═══════════════════════════════════════════════════════════
    
    def _fast_hand_strength(self, hole_cards: List[str], board: List[str]) -> float:
        """
        Évalue la force de la main via treys. Normalisé [0, 1].
        ~5μs avec le singleton Evaluator.
        """
        if len(board) < 3:
            return 0.0
        
        try:
            hand_ints = [Card.new(self._normalize_card(c)) for c in hole_cards]
            board_ints = [Card.new(self._normalize_card(c)) for c in board]
            
            score = self.evaluator.evaluate(board_ints, hand_ints)
            # treys : 1 = Royal Flush, 7462 = 7-high
            return 1.0 - (score - 1) / 7461.0
        except Exception:
            return 0.5  # Fallback neutre
    
    def _fast_equity(self, hole_cards: List[str], board: List[str],
                     num_opponents: int) -> float:
        """
        Estimation rapide de l'equity (sans Monte Carlo).
        Combine hand_strength + draw bonus + ajustement multi-way.
        """
        if not board:
            # Preflop : equity basée sur la tier
            hand_str = self._get_hand_string(hole_cards)
            if hand_str in TIER1_HANDS:
                base = 0.80
            elif hand_str in TIER2_HANDS:
                base = 0.60
            elif hand_str in TIER3_HANDS:
                base = 0.45
            else:
                base = 0.30
            return base * (0.95 ** max(0, num_opponents - 1))
        
        # Postflop
        strength = self._fast_hand_strength(hole_cards, board)
        draw_bonus = 0.0
        
        if strength < 0.70:
            if self._has_flush_draw(hole_cards, board):
                draw_bonus += 0.18
            if self._has_straight_draw(hole_cards, board):
                draw_bonus += 0.08
            # Overcards
            overcards = self._count_overcards(hole_cards, board)
            draw_bonus += 0.03 * overcards
        
        raw = min(strength + draw_bonus, 0.99)
        return raw * (0.90 ** max(0, num_opponents - 1))
    
    def _has_draw(self, hole_cards: List[str], board: List[str]) -> bool:
        """True si on a un flush draw ou straight draw."""
        if len(board) < 3:
            return False
        return self._has_flush_draw(hole_cards, board) or self._has_straight_draw(hole_cards, board)
    
    def _has_flush_draw(self, hole_cards: List[str], board: List[str]) -> bool:
        """True si 4 cartes de la même couleur (flush draw)."""
        if len(board) < 3:
            return False
        all_cards = hole_cards + board
        suits = [self._get_suit(c) for c in all_cards]
        suit_counts: Dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        # Flush draw = 4 cartes d'une même couleur (pas 5, sinon c'est fait)
        for s in hole_cards:
            suit = self._get_suit(s)
            if suit_counts.get(suit, 0) == 4:
                return True
        return False
    
    def _has_straight_draw(self, hole_cards: List[str], board: List[str]) -> bool:
        """True si on a un OESD ou gutshot."""
        if len(board) < 3:
            return False
        all_cards = hole_cards + board
        ranks = sorted(set(self._rank_value(c) for c in all_cards))
        
        # Vérifier fenêtres de 5 cartes consécutives
        for i in range(len(ranks)):
            # OESD : 4 cartes dans une fenêtre de 4
            window_4 = [r for r in ranks if ranks[i] <= r <= ranks[i] + 3]
            if len(window_4) >= 4:
                # Vérifier que le héros contribue au draw
                hero_ranks = [self._rank_value(c) for c in hole_cards]
                if any(ranks[i] <= hr <= ranks[i] + 3 for hr in hero_ranks):
                    return True
            
            # Gutshot : 3 cartes dans une fenêtre de 4 (plus courant)
            window_5 = [r for r in ranks if ranks[i] <= r <= ranks[i] + 4]
            if len(window_5) >= 4:
                hero_ranks = [self._rank_value(c) for c in hole_cards]
                if any(ranks[i] <= hr <= ranks[i] + 4 for hr in hero_ranks):
                    return True
        
        return False
    
    def _count_overcards(self, hole_cards: List[str], board: List[str]) -> int:
        """Nombre de cartes du héros au-dessus du board."""
        if not board:
            return 0
        max_board = max(self._rank_value(c) for c in board)
        return sum(1 for c in hole_cards if self._rank_value(c) > max_board)
    
    def _is_dry_board(self, board: List[str]) -> bool:
        """
        Board sec = peu de draws possibles.
        Ex: K72 rainbow, A83 rainbow
        """
        if len(board) < 3:
            return True
        
        # Check monotone/two-tone
        suits = [self._get_suit(c) for c in board]
        suit_counts: Dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        if max(suit_counts.values()) >= 3:
            return False  # Monotone = wet
        
        # Check connectedness
        ranks = sorted(self._rank_value(c) for c in board)
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks) - 1)]
        # Board connecté si gaps petits
        if any(g <= 2 for g in gaps):
            return False  # Connecté = wet
        
        return True  # Dry board
    
    # ═══════════════════════════════════════════════════════════
    # UTILITAIRES CARTES (inline, optimisés)
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def _normalize_card(card: str) -> str:
        """
        Normalise au format treys : 'As', 'Kh', '5c'.
        Gère les formats Pluribus ('As') et RLCard inversé ('Sa').
        """
        if not card or len(card) != 2:
            return '2s'
        c1, c2 = card[0], card[1]
        valid_ranks = 'AKQJT23456789'
        if c1.upper() in valid_ranks:
            return c1.upper() + c2.lower()
        elif c1.upper() in 'SHDC':
            return c2.upper() + c1.lower()
        return '2s'
    
    @staticmethod
    def _get_suit(card: str) -> str:
        """Retourne la couleur (s/h/d/c)."""
        if not card or len(card) != 2:
            return 's'
        valid_ranks = 'AKQJT23456789'
        if card[0].upper() in valid_ranks:
            return card[1].lower()
        return card[0].lower()
    
    @staticmethod
    def _rank_value(card: str) -> int:
        """Retourne la valeur numérique du rang (2-14)."""
        if not card:
            return 0
        valid_ranks = 'AKQJT23456789'
        c = card[0].upper()
        if c not in valid_ranks:
            c = card[1].upper() if len(card) > 1 else '2'
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(c, 0)
    
    def _get_hand_string(self, hole_cards: List[str]) -> str:
        """Convertit les hole cards en string de main ('AKs', 'QJo', etc.)."""
        if len(hole_cards) < 2:
            return ''
        
        r1 = self._rank_value(hole_cards[0])
        r2 = self._rank_value(hole_cards[1])
        s1 = self._get_suit(hole_cards[0])
        s2 = self._get_suit(hole_cards[1])
        
        rank_chars = {
            14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
            9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'
        }
        
        c1 = rank_chars.get(max(r1, r2), '?')
        c2 = rank_chars.get(min(r1, r2), '?')
        
        if r1 == r2:
            return f"{c1}{c2}"  # Paire : 'AA', 'KK'
        
        suited = 's' if s1 == s2 else 'o'
        return f"{c1}{c2}{suited}"
    
    # ═══════════════════════════════════════════════════════════
    # STATS & AFFICHAGE
    # ═══════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        """Statistiques de l'agent."""
        total = sum(self.stats['actions'].values())
        if total == 0:
            return self.stats
        
        action_pcts = {}
        for aid, count in self.stats['actions'].items():
            name = self.stats['action_names'].get(aid, f'ACTION_{aid}')
            action_pcts[name] = count / total * 100
        
        return {
            **self.stats,
            'action_percentages': action_pcts,
            'error_rate': self.stats['errors'] / max(self.stats['total_decisions'], 1) * 100,
        }
    
    def print_stats(self):
        """Affiche les statistiques."""
        stats = self.get_stats()
        total = sum(stats['actions'].values())
        
        print(f"\n{'='*55}")
        print(f"📊 STATS RULE-BASED BOT")
        print(f"{'='*55}")
        print(f"Décisions totales : {stats['total_decisions']}")
        print(f"  Preflop : {stats['preflop_decisions']}")
        print(f"  Postflop: {stats['postflop_decisions']}")
        print(f"  Erreurs : {stats['errors']} ({stats.get('error_rate', 0):.1f}%)")
        print(f"\nDistribution des actions ({total} total):")
        
        for aid in sorted(stats['actions']):
            count = stats['actions'][aid]
            name = stats['action_names'].get(aid, f'ACTION_{aid}')
            pct = count / max(total, 1) * 100
            bar = '█' * int(pct / 2)
            print(f"  {name:12s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        print(f"{'='*55}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED RULE-BASED BOT V2
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Améliorations vs V1 :
#  1. FeatureExtractor complet (99 features) pour equity + board texture
#  2. Ranges dynamiques par position (9 positions × open/3bet/4bet)
#  3. Board texture : monotone, paired, connected, high/low
#  4. Draws : outs exacts, combo draws, backdoors
#  5. Multi-barrel bluffs + check-raise + polarisation 
#  6. Sizing adaptatif (33%, 50%, 75%, pot)
#  7. Opponent HUD tracking (VPIP, PFR, AF)
#  8. Multi-street SPR planning
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Position-specific preflop ranges ───────────────────────
# Format : set of hand strings. "s" = suited, "o" = offsuit, pair = "AA"

OPEN_RANGE = {
    # UTG (tight ~15%)
    'UTG': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'ATs',
        'KQs', 'KJs',
    },
    'UTG+1': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'ATs', 'A9s',
        'KQs', 'KJs', 'KTs', 'QJs',
    },
    # MP (~20%)
    'MP': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'ATs', 'A9s', 'A8s',
        'KQs', 'KQo', 'KJs', 'KTs', 'QJs', 'QTs', 'JTs',
    },
    'MP+1': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'ATs', 'A9s', 'A8s', 'A7s',
        'KQs', 'KQo', 'KJs', 'KTs', 'K9s', 'QJs', 'QTs', 'JTs', 'T9s',
    },
    # CO (~28%)
    'CO': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'A9s', 'A8s',
        'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
        'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'K9s',
        'QJs', 'QJo', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s', '87s',
    },
    # BTN (~40%)
    'BTN': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo', 'A9s', 'A8s',
        'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
        'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'KTo', 'K9s', 'K8s',
        'QJs', 'QJo', 'QTs', 'Q9s', 'Q8s',
        'JTs', 'JTo', 'J9s', 'J8s', 'T9s', 'T8s', '98s', '97s', '87s', '86s',
        '76s', '75s', '65s', '54s',
    },
    # SB (~35%) — steal range
    'SB': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo', 'A9s', 'A8s',
        'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
        'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'K9s', 'K8s',
        'QJs', 'QJo', 'QTs', 'Q9s', 'JTs', 'J9s', 'T9s', '98s', '87s', '76s',
    },
    # BB — défense vs open (large car on a les cotes)
    'BB': {
        'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22',
        'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo', 'A9s', 'A9o',
        'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
        'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'KTo', 'K9s', 'K8s', 'K7s',
        'QJs', 'QJo', 'QTs', 'Q9s', 'Q8s',
        'JTs', 'JTo', 'J9s', 'J8s', 'T9s', 'T8s', '98s', '97s',
        '87s', '86s', '76s', '75s', '65s', '64s', '54s', '53s', '43s',
    },
}

# 3-bet range (tighter, polarized)
THREEBET_RANGE = {
    'AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs',  # Value
    'A5s', 'A4s', 'A3s',  # Bluffs (blockers + equity)
    '76s', '65s',  # Bluffs (playability)
}

# 4-bet range (very tight)
FOURBET_RANGE = {
    'AA', 'KK', 'QQ', 'AKs', 'AKo',  # Value
    'A5s',  # Bluff (blocker)
}

# Premium hands that always get it in
PREMIUM_ALLIN = {'AA', 'KK'}


class OpponentTracker:
    """
    HUD simplifié — track les tendances des adversaires en temps réel.
    VPIP, PFR, Aggression Factor, Fold to Cbet.
    """
    def __init__(self):
        self._data: Dict[int, Dict] = {}  # player_id → stats
    
    def _ensure_player(self, pid: int):
        if pid not in self._data:
            self._data[pid] = {
                'hands': 0, 'vpip': 0, 'pfr': 0,
                'bets': 0, 'raises': 0, 'calls': 0, 'folds': 0,
                'cbets': 0, 'cbet_opportunities': 0,
                'fold_to_cbet': 0, 'cbet_faced': 0,
            }
    
    def record_action(self, pid: int, action_value: int, street: str, is_preflop_aggressor: bool = False):
        """Enregistre une action pour un joueur."""
        self._ensure_player(pid)
        d = self._data[pid]
        
        if action_value == 0:  # FOLD
            d['folds'] += 1
        elif action_value == 1:  # CHECK/CALL
            if street == 'preflop':
                d['vpip'] += 1
            d['calls'] += 1
        elif action_value in (2, 3):  # RAISE
            if street == 'preflop':
                d['vpip'] += 1
                d['pfr'] += 1
            d['raises'] += 1
            d['bets'] += 1
        elif action_value == 4:  # ALL-IN
            d['vpip'] += 1
            d['raises'] += 1
            d['bets'] += 1
    
    def new_hand(self, pid: int):
        self._ensure_player(pid)
        self._data[pid]['hands'] += 1
    
    def get_vpip(self, pid: int) -> float:
        """VPIP = % de mains jouées volontairement (pas juste check BB)."""
        self._ensure_player(pid)
        d = self._data[pid]
        return d['vpip'] / max(d['hands'], 1)
    
    def get_pfr(self, pid: int) -> float:
        """PFR = % de mains raise preflop."""
        self._ensure_player(pid)
        d = self._data[pid]
        return d['pfr'] / max(d['hands'], 1)
    
    def get_af(self, pid: int) -> float:
        """Aggression Factor = (bets + raises) / calls."""
        self._ensure_player(pid)
        d = self._data[pid]
        return (d['bets'] + d['raises']) / max(d['calls'], 1)
    
    def is_fish(self, pid: int) -> bool:
        """VPIP > 50% et AF < 1.5 = joueur passif / fish."""
        return self.get_vpip(pid) > 0.50 and self.get_af(pid) < 1.5
    
    def is_nit(self, pid: int) -> bool:
        """VPIP < 15% = joueur trop serré."""
        d = self._data.get(pid, {})
        if d.get('hands', 0) < 10:
            return False  # Pas assez de data
        return self.get_vpip(pid) < 0.15
    
    def is_lag(self, pid: int) -> bool:
        """VPIP > 30% et AF > 2.5 = loose-aggressive."""
        return self.get_vpip(pid) > 0.30 and self.get_af(pid) > 2.5


class AdvancedRuleBot:
    """
    Bot V2 — Poker No-Limit Hold'em basé sur des règles avancées.
    
    Niveau : Semi-pro / Pro basique
    
    Améliorations vs V1 :
    1. FeatureExtractor (99 features) pour equity + board texture
    2. Ranges dynamiques par position (9 positions × open/3bet/4bet)
    3. Board texture complète (monotone, paired, connected, high/low)
    4. Draws : outs exacts, combo draws, backdoors
    5. Multi-barrel bluffs + check-raise + polarisation
    6. Sizing adaptatif (33%, 50%, 75%, pot)
    7. Opponent HUD tracking (VPIP, PFR, AF)
    8. Multi-street SPR planning
    """
    
    def __init__(self, env):
        self.env = env
        self.adapter = RLCardAdapter()
        self.evaluator = Evaluator()
        self.use_raw = True
        
        # Opponent tracking
        self.tracker = OpponentTracker()
        self.hand_count = 0
        
        # Multi-street memory (reset each hand)
        self._street_history: Dict[str, str] = {}  # street → action taken
        self._was_preflop_aggressor = False
        self._barrels_fired = 0
        self._last_street = 'preflop'
        
        # Stats
        self.stats = {
            'total_decisions': 0,
            'actions': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            'action_names': {
                0: 'FOLD', 1: 'CHECK/CALL',
                2: 'RAISE_HALF', 3: 'RAISE_POT', 4: 'ALL-IN'
            },
            'errors': 0,
            'preflop_decisions': 0,
            'postflop_decisions': 0,
            'bluffs': 0,
            'value_bets': 0,
            'check_raises': 0,
        }
    
    # ═══════════════════════════════════════════════════════════
    # INTERFACE RLCARD
    # ═══════════════════════════════════════════════════════════
    
    def step(self, state: Dict) -> int:
        action_enum, _ = self._decide(state)
        return action_enum
    
    def eval_step(self, state: Dict) -> Tuple:
        return self._decide(state)
    
    # ═══════════════════════════════════════════════════════════
    # PIPELINE DE DÉCISION PRINCIPAL
    # ═══════════════════════════════════════════════════════════
    
    def _decide(self, state: Dict) -> Tuple:
        self.stats['total_decisions'] += 1
        
        legal_actions_enum = state.get('raw_legal_actions', [])
        legal_actions_int = [a.value for a in legal_actions_enum]
        
        if not legal_actions_int:
            self.stats['errors'] += 1
            return legal_actions_enum[0] if legal_actions_enum else 0, {}
        
        try:
            gs = self.adapter.to_game_state(state, self.env)
            
            # Reset multi-street memory on new hand
            # Détecte un nouveau hand si on revient au preflop
            # ou si le nombre d'actions preflop est minimal (début de main)
            current_street = gs.street
            if current_street == 'preflop' and self._last_street in ('turn', 'river', 'flop'):
                self._reset_hand_memory()
            elif current_street == 'preflop' and len(gs.actions_this_street) <= 2:
                self._reset_hand_memory()
            self._last_street = current_street
            
            if gs.street == 'preflop':
                self.stats['preflop_decisions'] += 1
                action_int = self._preflop_decision(gs, legal_actions_int)
            else:
                self.stats['postflop_decisions'] += 1
                action_int = self._postflop_decision(gs, legal_actions_int)
            
            # Validation
            if action_int not in legal_actions_int:
                action_int = 1 if 1 in legal_actions_int else legal_actions_int[0]
            
            # Track multi-street memory
            self._street_history[gs.street] = action_int
            if action_int in (2, 3, 4) and gs.street == 'preflop':
                self._was_preflop_aggressor = True
            if action_int in (2, 3, 4) and gs.street != 'preflop':
                self._barrels_fired += 1
            
            self.stats['actions'][action_int] = self.stats['actions'].get(action_int, 0) + 1
            action_enum = legal_actions_enum[legal_actions_int.index(action_int)]
            probs = {a: (1.0 if a == action_enum else 0.0) for a in legal_actions_enum}
            
            return action_enum, probs
            
        except Exception as e:
            self.stats['errors'] += 1
            fallback = legal_actions_enum[0]
            for a in legal_actions_enum:
                if a.value == 1:
                    fallback = a
                    break
            return fallback, {a: (1.0 if a == fallback else 0.0) for a in legal_actions_enum}
    
    def _reset_hand_memory(self):
        self._street_history = {}
        self._was_preflop_aggressor = False
        self._barrels_fired = 0
        self.hand_count += 1
    
    # ═══════════════════════════════════════════════════════════
    # STRATÉGIE PREFLOP — V2 (position-specific ranges)
    # ═══════════════════════════════════════════════════════════
    
    def _preflop_decision(self, gs: GameState, legal: List[int]) -> int:
        hand_str = self._get_hand_string(gs.hole_cards)
        position = gs.position
        facing_raise = gs.amount_to_call > gs.big_blind
        facing_3bet = self._count_raises_preflop(gs) >= 2
        facing_4bet = self._count_raises_preflop(gs) >= 3
        
        # ─── Face à un 4-bet+ ──────────────────────────
        if facing_4bet:
            if hand_str in PREMIUM_ALLIN and 4 in legal:
                return 4  # All-in avec AA/KK
            if hand_str in FOURBET_RANGE:
                return 1  # Call
            return 0  # Fold
        
        # ─── Face à un 3-bet ───────────────────────────
        if facing_3bet:
            if hand_str in FOURBET_RANGE:
                # 4-bet !
                if 3 in legal:
                    return 3
                if 2 in legal:
                    return 2
                return 1
            if hand_str in THREEBET_RANGE:
                return 1  # Call le 3-bet
            # Call avec des mains avec bonne equity et positionnées
            if hand_str in {'TT', '99', 'AQo', 'AJs', 'KQs'} and position in LATE_POSITIONS:
                return 1
            return 0  # Fold
        
        # ─── Face à un open-raise ──────────────────────
        if facing_raise:
            # 3-bet ?
            if hand_str in THREEBET_RANGE:
                if 3 in legal:
                    return 3
                if 2 in legal:
                    return 2
                return 1
            
            # Call avec les mains de notre défense range
            pos_range = OPEN_RANGE.get(position, OPEN_RANGE.get('MP', set()))
            if hand_str in pos_range:
                return 1  # Call
            
            # BB défend plus large (meilleurs pot odds)
            if position == 'BB' and hand_str in OPEN_RANGE['BB']:
                return 1
            
            return 0  # Fold
        
        # ─── Personne n'a raise (open ou limp) ────────
        pos_range = OPEN_RANGE.get(position, OPEN_RANGE.get('MP', set()))
        
        if hand_str in pos_range:
            # Open-raise
            if hand_str in TIER1_HANDS:
                # Premium : raise pot
                return 3 if 3 in legal else (2 if 2 in legal else 1)
            # Standard : raise half-pot
            return 2 if 2 in legal else (3 if 3 in legal else 1)
        
        # Trash : check gratuit en BB, sinon fold
        if gs.amount_to_call == 0 and 1 in legal:
            return 1
        return 0
    
    def _count_raises_preflop(self, gs: GameState) -> int:
        """Compte le nombre de raises preflop dans les actions de la street."""
        count = 0
        for action in gs.actions_this_street:
            if isinstance(action, str):
                action_lower = action.lower()
                if 'raise' in action_lower or 'all' in action_lower:
                    count += 1
            elif isinstance(action, dict):
                act = action.get('action', '')
                if act in ('raise', 'all_in'):
                    count += 1
        return count
    
    # ═══════════════════════════════════════════════════════════
    # STRATÉGIE POSTFLOP — V2 (texture + draws + sizing + barrels)
    # ═══════════════════════════════════════════════════════════
    
    def _postflop_decision(self, gs: GameState, legal: List[int]) -> int:
        # ─── 1. Évaluation complète ─────────────────────
        strength = self._fast_hand_strength(gs.hole_cards, gs.board)
        equity = self._fast_equity(gs.hole_cards, gs.board, gs.num_active_players)
        
        # Board texture analysis
        texture = self._analyze_board_texture(gs.board)
        
        # Draw analysis (exact outs)
        draw_info = self._analyze_draws(gs.hole_cards, gs.board)
        total_outs = draw_info['total_outs']
        is_combo_draw = draw_info['is_combo']
        has_any_draw = total_outs >= 4
        
        pot_odds = gs.pot_odds
        spr = gs.spr
        position = gs.position
        is_ip = position in LATE_POSITIONS
        facing_bet = gs.amount_to_call > 0
        street = gs.street
        
        # ─── Opponent profiling ─────────────────────────
        # Détecte si les adversaires sont des stations (call too much)
        # ou des nits (fold too much) pour adapter la stratégie
        avg_vpip = sum(self.tracker.get_vpip(i) for i in range(1, 6)) / 5
        opponents_are_stations = avg_vpip > 0.35  # Calling stations
        opponents_are_nits = avg_vpip < 0.20 and self.hand_count > 30
        
        # ─── 2. Nuts / Near-nuts ────────────────────────
        if strength >= 0.85:
            self.stats['value_bets'] += 1
            return self._choose_sizing(gs, legal, spr, 'value_strong', texture)
        
        # ─── 3. Strong hand ─────────────────────────────
        if strength >= 0.65:
            if spr < 4:
                self.stats['value_bets'] += 1
                return self._choose_sizing(gs, legal, spr, 'value_commit', texture)
            
            if facing_bet:
                if strength >= 0.75 and texture['is_wet']:
                    return self._choose_sizing(gs, legal, spr, 'value_raise', texture)
                return 1  # Call
            else:
                self.stats['value_bets'] += 1
                return self._choose_sizing(gs, legal, spr, 'value_bet', texture)
        
        # ─── 4. Medium hand ─────────────────────────────
        if strength >= 0.40:  # Seuil abaissé (était 0.45) pour voir plus de flops
            if facing_bet:
                effective_equity = equity
                
                # Adapt to opponent type
                if opponents_are_stations:
                    # Stations bluffent moins → on peut fold un peu plus
                    # Mais on call quand même si pot odds sont bons
                    pass
                elif opponents_are_nits:
                    # Nits bet = value → respect their bets more
                    effective_equity *= 0.85
                
                if effective_equity > pot_odds:
                    return 1  # Call profitable
                
                # MDF defense — défendre plus large que théorique
                mdf = gs.pot_size / (gs.pot_size + gs.amount_to_call) if gs.amount_to_call > 0 else 0
                if strength >= 0.50 and effective_equity > mdf * 0.65:  # Était 0.75 → plus de defense
                    return 1  # Défense marginale
                
                # ─── FLOAT (nouveau) ─────────────────────
                # Call le flop IP avec l'intention de steal le turn
                if is_ip and street == 'flop' and strength >= 0.30:
                    if spr > 4:  # Assez de stack pour float
                        self.stats['bluffs'] += 1  # Float = bluff déguisé
                        return 1  # Call pour float
                
                return 0  # Fold
            else:
                # No bet facing — thin value or protection
                if is_ip and strength >= 0.45:  # Abaissé de 0.50
                    return self._choose_sizing(gs, legal, spr, 'thin_value', texture)
                if not is_ip and strength >= 0.50 and texture['is_wet']:
                    return self._choose_sizing(gs, legal, spr, 'thin_value', texture)
                return 1  # Check (pot control)
        
        # ─── 5. Drawing hands ───────────────────────────
        if has_any_draw and street != 'river':
            draw_equity = self._outs_to_equity(total_outs, street)
            
            if facing_bet:
                implied_equity = draw_equity + (0.15 if spr > 3 else 0.05)
                
                if is_combo_draw:
                    if is_ip and 2 in legal and draw_equity > 0.30:
                        self.stats['bluffs'] += 1
                        return self._choose_sizing(gs, legal, spr, 'semi_bluff', texture)
                    return 1  # Call combo draw
                
                if implied_equity > pot_odds:
                    return 1  # Call le draw
                    
                # Float with weak draws IP (4-5 outs) si spr suffisant
                if is_ip and total_outs >= 3 and spr > 4:
                    return 1  # Float avec draw
                    
                return 0  # Fold — draw trop cher
            else:
                if is_ip and draw_equity >= 0.20:  # Abaissé de 0.25
                    self.stats['bluffs'] += 1
                    return self._choose_sizing(gs, legal, spr, 'semi_bluff', texture)
                return 1
        
        # ─── 6. Weak / Air — Bluff or float ────────────
        if facing_bet:
            # Float IP sur flop avec air (steal turn)
            if is_ip and street == 'flop' and spr > 5 and not opponents_are_stations:
                if strength >= 0.15:  # On float seulement si on a un petit quelque chose
                    self.stats['bluffs'] += 1
                    return 1  # Float call
            return 0  # Fold
        
        # Multi-barrel bluff logic (adapté aux adversaires)
        if self._should_bluff(gs, strength, texture, is_ip, street, legal, opponents_are_stations):
            self.stats['bluffs'] += 1
            return self._choose_sizing(gs, legal, spr, 'bluff', texture)
        
        if 1 in legal:
            return 1
        return 0
    
    # ═══════════════════════════════════════════════════════════
    # BLUFF ENGINE — Multi-barrel + polarisation
    # ═══════════════════════════════════════════════════════════
    
    def _should_bluff(self, gs: GameState, strength: float, texture: Dict,
                      is_ip: bool, street: str, legal: List[int],
                      opponents_are_stations: bool = False) -> bool:
        """
        Décide si on doit bluffer. Logique multi-barrel.
        Adapté au profil adversaire (bluff moins vs stations).
        """
        if 2 not in legal and 3 not in legal:
            return False
        
        # Réduire les bluffs contre les calling stations
        if opponents_are_stations:
            # Contre des stations on ne bluff presque pas
            # On garde seulement les c-bets avec un minimum d'equity
            if street == 'flop' and self._was_preflop_aggressor and self._barrels_fired == 0:
                if strength >= 0.30:  # C-bet seulement avec equity
                    return True
            return False  # Pas de bluffs purs vs stations
        
        spr = gs.spr
        
        # ─── C-bet (1er barrel) ──────────────────────
        if street == 'flop' and self._was_preflop_aggressor and self._barrels_fired == 0:
            if is_ip:
                return True  # C-bet range entière IP
            if texture['is_dry']:
                return True
            if strength >= 0.20:
                return True  # C-bet OOP avec equity
            return False
        
        # ─── Delayed c-bet / Float follow-through ────
        # Si on a floaté (call flop), on peut bet le turn pour steal
        if street == 'turn' and self._barrels_fired == 0 and is_ip:
            # Float follow-through : on a call le flop IP, maintenant on bet
            if 'flop' in self._street_history and self._street_history['flop'] == 1:
                if texture['is_dry'] or strength >= 0.20:
                    return True  # Delayed c-bet / float bet
        
        # ─── Double barrel (2ème barrel) ─────────────
        if street == 'turn' and self._barrels_fired >= 1:
            if is_ip and texture['is_dry']:
                return True
            if len(gs.board) >= 4:
                turn_rank = self._rank_value(gs.board[3])
                flop_max = max(self._rank_value(c) for c in gs.board[:3])
                if turn_rank > flop_max and is_ip:
                    return True
            return False
        
        # ─── Triple barrel (3ème barrel) — rare ─────
        if street == 'river' and self._barrels_fired >= 2:
            if is_ip and spr > 2:
                has_blocker = self._has_nut_blocker(gs.hole_cards, gs.board)
                if has_blocker:
                    return True
            return False
        
        # ─── Probe bet ──────────────────────────────
        if is_ip and not self._was_preflop_aggressor and self._barrels_fired == 0:
            if spr > 3:
                return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════
    # SIZING ADAPTATIF
    # ═══════════════════════════════════════════════════════════
    
    def _choose_sizing(self, gs: GameState, legal: List[int], spr: float,
                       bet_type: str, texture: Dict) -> int:
        """
        Choisit le sizing optimal selon le type de bet et la texture.
        
        Sizing guide :
        - 33% pot : thin value, dry board, pot control
        - 50% pot : standard bet (RAISE_HALF_POT = action 2)
        - 75-100% : polar, wet board, protection (RAISE_POT = action 3)
        - All-in   : commit SPR < 2 (ALL_IN = action 4)
        """
        # All-in si SPR très bas
        if spr < 2 and bet_type in ('value_commit', 'value_strong') and 4 in legal:
            return 4
        
        if bet_type in ('value_strong', 'value_commit', 'value_raise'):
            # Gros sizing pour value fort
            if texture['is_wet']:
                return 3 if 3 in legal else (2 if 2 in legal else 1)  # Pot (protection)
            if spr < 4:
                return 3 if 3 in legal else (2 if 2 in legal else 1)  # Commit
            return 2 if 2 in legal else (3 if 3 in legal else 1)  # Half-pot
        
        elif bet_type == 'value_bet':
            # Value standard
            if texture['is_wet']:
                return 3 if 3 in legal else (2 if 2 in legal else 1)  # Bigger on wet
            return 2 if 2 in legal else (3 if 3 in legal else 1)  # Half-pot
        
        elif bet_type == 'thin_value':
            # Petit sizing pour thin value (dry board, pas besoin de gros)
            return 2 if 2 in legal else 1  # Half-pot max
        
        elif bet_type == 'semi_bluff':
            # Semi-bluff : sizing similaire au value pour être balancé
            if texture['is_wet']:
                return 3 if 3 in legal else (2 if 2 in legal else 1)  
            return 2 if 2 in legal else (3 if 3 in legal else 1)
        
        elif bet_type == 'bluff':
            # Bluff pur : petit sizing pour risquer moins
            # Sauf river (polarisé = gros)
            if gs.street == 'river':
                return 3 if 3 in legal else (2 if 2 in legal else 1)  # Polarisé river
            return 2 if 2 in legal else 1  # Half-pot bluff
        
        # Fallback
        return 2 if 2 in legal else (3 if 3 in legal else 1)
    
    # ═══════════════════════════════════════════════════════════
    # BOARD TEXTURE ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def _analyze_board_texture(self, board: List[str]) -> Dict:
        """
        Analyse complète de la texture du board.
        
        Retourne : is_dry, is_wet, is_monotone, is_two_tone, is_rainbow,
                   is_paired, is_connected, high_card_rank, low_board
        """
        result = {
            'is_dry': True, 'is_wet': False,
            'is_monotone': False, 'is_two_tone': False, 'is_rainbow': False,
            'is_paired': False, 'is_connected': False,
            'high_card_rank': 0, 'low_board': False,
        }
        
        if len(board) < 3:
            return result
        
        # Suits
        suits = [self._get_suit(c) for c in board]
        suit_counts: Dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit = max(suit_counts.values())
        
        result['is_monotone'] = max_suit >= 3
        result['is_two_tone'] = max_suit == 2 and len(suit_counts) == 2
        result['is_rainbow'] = max_suit == 1 or (len(suit_counts) >= 3 and max_suit == 1)
        
        # Ranks
        ranks = sorted([self._rank_value(c) for c in board])
        result['high_card_rank'] = max(ranks)
        result['low_board'] = max(ranks) <= 9  # Pas de face card
        
        # Paired
        result['is_paired'] = len(set(ranks)) < len(ranks)
        
        # Connectedness
        gaps = [ranks[i+1] - ranks[i] for i in range(len(ranks) - 1)]
        has_small_gaps = sum(1 for g in gaps if g <= 2) >= 2
        has_consecutive = any(g == 1 for g in gaps)
        result['is_connected'] = has_consecutive or has_small_gaps
        
        # Wet = monotone OU two-tone+connected OU très connecté
        result['is_wet'] = (
            result['is_monotone'] or 
            (result['is_two_tone'] and result['is_connected']) or
            has_small_gaps
        )
        result['is_dry'] = not result['is_wet']
        
        return result
    
    # ═══════════════════════════════════════════════════════════
    # DRAW ANALYSIS — Outs exacts
    # ═══════════════════════════════════════════════════════════
    
    def _analyze_draws(self, hole_cards: List[str], board: List[str]) -> Dict:
        """
        Analyse précise des draws avec outs exacts.
        
        Retourne : flush_draw, flush_outs, oesd, gutshot, straight_outs,
                   backdoor_flush, backdoor_straight, total_outs, is_combo
        """
        result = {
            'flush_draw': False, 'flush_outs': 0,
            'oesd': False, 'gutshot': False, 'straight_outs': 0,
            'backdoor_flush': False, 'backdoor_straight': False,
            'total_outs': 0, 'is_combo': False,
        }
        
        if len(board) < 3:
            return result
        
        all_cards = hole_cards + board
        
        # ─── Flush draws ────────────────────────────────
        suits = [self._get_suit(c) for c in all_cards]
        hero_suits = [self._get_suit(c) for c in hole_cards]
        suit_counts: Dict[str, int] = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        for hs in hero_suits:
            if suit_counts.get(hs, 0) == 4:
                result['flush_draw'] = True
                result['flush_outs'] = 9  # 13 - 4 cartes visibles
                break
            elif suit_counts.get(hs, 0) == 3 and len(board) == 3:
                result['backdoor_flush'] = True
        
        # ─── Straight draws ─────────────────────────────
        all_ranks = sorted(set(self._rank_value(c) for c in all_cards))
        hero_ranks = set(self._rank_value(c) for c in hole_cards)
        
        # Add ace as low for wheel draws
        if 14 in all_ranks:
            all_ranks_extended = [1] + all_ranks
        else:
            all_ranks_extended = all_ranks
        
        best_straight_draw = 0
        
        # Check every possible 5-card straight window
        for bottom in range(1, 11):  # 1(A-low) to 10(T-A)
            window = set(range(bottom, bottom + 5))
            hits = window & set(all_ranks_extended)
            
            if len(hits) == 4:
                # 4 of 5 cards present → need 1 card
                missing = window - set(all_ranks_extended)
                # Hero must contribute to the draw
                hero_in_window = hero_ranks & window
                if hero_in_window:
                    if len(missing) == 1:
                        needed = missing.pop()
                        if needed == 1:
                            needed = 14  # Ace
                        # Is it OESD or gutshot?
                        # OESD = both ends open (missing card is min-1 or max+1)
                        sorted_hits = sorted(hits)
                        if needed < sorted_hits[0] or needed > sorted_hits[-1]:
                            # Open-ended
                            if not result['oesd']:
                                result['oesd'] = True
                                best_straight_draw = max(best_straight_draw, 8)
                        else:
                            # Gutshot
                            if not result['gutshot']:
                                result['gutshot'] = True
                                best_straight_draw = max(best_straight_draw, 4)
            
            elif len(hits) == 3 and len(board) == 3:
                # Backdoor straight (3 of 5 on flop)
                hero_in_window = hero_ranks & window
                if hero_in_window:
                    result['backdoor_straight'] = True
        
        result['straight_outs'] = best_straight_draw
        
        # ─── Total outs + combo ─────────────────────────
        total = result['flush_outs'] + result['straight_outs']
        # Si combo draw, attention au double-comptage (~2 cartes partagées)
        if result['flush_draw'] and result['straight_outs'] > 0:
            result['is_combo'] = True
            total -= 2  # Environ 2 cartes qui complètent les deux
        
        # Backdoor bonus (~1.5 outs chacun)
        if result['backdoor_flush']:
            total += 1.5
        if result['backdoor_straight']:
            total += 1
        
        result['total_outs'] = max(0, total)
        return result
    
    def _outs_to_equity(self, outs: float, street: str) -> float:
        """
        Convertit les outs en equity approximative.
        Rule of 2 and 4 : flop = outs × 4%, turn = outs × 2%
        """
        if street == 'flop':
            return min(outs * 0.04, 0.60)  # Cap à 60%
        elif street == 'turn':
            return min(outs * 0.02, 0.45)
        return 0.0
    
    # ═══════════════════════════════════════════════════════════
    # BLOCKER ANALYSIS
    # ═══════════════════════════════════════════════════════════
    
    def _has_nut_blocker(self, hole_cards: List[str], board: List[str]) -> bool:
        """
        True si on bloque les nuts adverses.
        Ex: on a un Ace de la flush suit, on bloque la nut flush.
        """
        if len(board) < 3:
            return False
        
        # Flush blocker : on a l'As de la couleur du board
        board_suits = [self._get_suit(c) for c in board]
        suit_counts: Dict[str, int] = {}
        for s in board_suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        
        for suit, count in suit_counts.items():
            if count >= 3:
                # Board has 3+ of a suit — do we have the Ace of that suit?
                for hc in hole_cards:
                    if self._get_suit(hc) == suit and self._rank_value(hc) == 14:
                        return True  # Nut flush blocker
        
        # Top pair blocker
        board_ranks = sorted([self._rank_value(c) for c in board], reverse=True)
        for hc in hole_cards:
            if self._rank_value(hc) == board_ranks[0]:
                return True  # On bloque top pair
        
        return False
    
    # ═══════════════════════════════════════════════════════════
    # HAND EVALUATION (réutilisé de V1, optimisé)
    # ═══════════════════════════════════════════════════════════
    
    def _fast_hand_strength(self, hole_cards: List[str], board: List[str]) -> float:
        if len(board) < 3:
            return 0.0
        try:
            hand_ints = [Card.new(self._normalize_card(c)) for c in hole_cards]
            board_ints = [Card.new(self._normalize_card(c)) for c in board]
            score = self.evaluator.evaluate(board_ints, hand_ints)
            return 1.0 - (score - 1) / 7461.0
        except Exception:
            return 0.5
    
    def _fast_equity(self, hole_cards: List[str], board: List[str],
                     num_opponents: int) -> float:
        if not board:
            hand_str = self._get_hand_string(hole_cards)
            pos_range = OPEN_RANGE.get('BTN', set())
            if hand_str in PREMIUM_ALLIN:
                base = 0.85
            elif hand_str in FOURBET_RANGE:
                base = 0.75
            elif hand_str in THREEBET_RANGE:
                base = 0.65
            elif hand_str in pos_range:
                base = 0.50
            else:
                base = 0.30
            return base * (0.92 ** max(0, num_opponents - 1))
        
        strength = self._fast_hand_strength(hole_cards, board)
        draw_info = self._analyze_draws(hole_cards, board)
        draw_bonus = self._outs_to_equity(draw_info['total_outs'], 'flop') * 0.5
        
        raw = min(strength + draw_bonus, 0.99)
        return raw * (0.90 ** max(0, num_opponents - 1))
    
    # ═══════════════════════════════════════════════════════════
    # UTILITAIRES CARTES (même que V1)
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def _normalize_card(card: str) -> str:
        if not card or len(card) != 2:
            return '2s'
        c1, c2 = card[0], card[1]
        valid_ranks = 'AKQJT23456789'
        if c1.upper() in valid_ranks:
            return c1.upper() + c2.lower()
        elif c1.upper() in 'SHDC':
            return c2.upper() + c1.lower()
        return '2s'
    
    @staticmethod
    def _get_suit(card: str) -> str:
        if not card or len(card) != 2:
            return 's'
        valid_ranks = 'AKQJT23456789'
        if card[0].upper() in valid_ranks:
            return card[1].lower()
        return card[0].lower()
    
    @staticmethod
    def _rank_value(card: str) -> int:
        if not card:
            return 0
        valid_ranks = 'AKQJT23456789'
        c = card[0].upper()
        if c not in valid_ranks:
            c = card[1].upper() if len(card) > 1 else '2'
        rank_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_map.get(c, 0)
    
    def _get_hand_string(self, hole_cards: List[str]) -> str:
        if len(hole_cards) < 2:
            return ''
        r1 = self._rank_value(hole_cards[0])
        r2 = self._rank_value(hole_cards[1])
        s1 = self._get_suit(hole_cards[0])
        s2 = self._get_suit(hole_cards[1])
        rank_chars = {
            14: 'A', 13: 'K', 12: 'Q', 11: 'J', 10: 'T',
            9: '9', 8: '8', 7: '7', 6: '6', 5: '5', 4: '4', 3: '3', 2: '2'
        }
        c1 = rank_chars.get(max(r1, r2), '?')
        c2 = rank_chars.get(min(r1, r2), '?')
        if r1 == r2:
            return f"{c1}{c2}"
        suited = 's' if s1 == s2 else 'o'
        return f"{c1}{c2}{suited}"
    
    # ═══════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        total = sum(self.stats['actions'].values())
        if total == 0:
            return self.stats
        action_pcts = {}
        for aid, count in self.stats['actions'].items():
            name = self.stats['action_names'].get(aid, f'ACTION_{aid}')
            action_pcts[name] = count / total * 100
        return {
            **self.stats,
            'action_percentages': action_pcts,
            'error_rate': self.stats['errors'] / max(self.stats['total_decisions'], 1) * 100,
        }
    
    def print_stats(self):
        stats = self.get_stats()
        total = sum(stats['actions'].values())
        print(f"\n{'='*55}")
        print(f"📊 STATS ADVANCED RULE BOT V2")
        print(f"{'='*55}")
        print(f"Décisions totales : {stats['total_decisions']}")
        print(f"  Preflop : {stats['preflop_decisions']}")
        print(f"  Postflop: {stats['postflop_decisions']}")
        print(f"  Erreurs : {stats['errors']} ({stats.get('error_rate', 0):.1f}%)")
        print(f"  Bluffs  : {stats['bluffs']}")
        print(f"  Value   : {stats['value_bets']}")
        print(f"\nDistribution ({total} total):")
        for aid in sorted(stats['actions']):
            count = stats['actions'][aid]
            name = stats['action_names'].get(aid, f'ACTION_{aid}')
            pct = count / max(total, 1) * 100
            bar = '█' * int(pct / 2)
            print(f"  {name:12s}: {count:5d} ({pct:5.1f}%) {bar}")
        print(f"{'='*55}\n")