from typing import List, Dict, Tuple
import numpy as np
from core.game_state import GameState
from features.card_features import UtilsCardFeatures

class GameTheoryFeatures(UtilsCardFeatures):
    def __init__(self, premium_hands=None, strong_hands=None, evaluator=None):
        super().__init__(premium_hands, strong_hands)
        self.evaluator = evaluator

    def _extract_game_theory_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        
        # EV estimations améliorées (3)
        equity = self._estimate_equity(state.hole_cards, state.board, state.num_active_players)
        pot_odds = state.pot_odds
        
        pot_total = state.pot_size + state.amount_to_call
        ev_call_raw = equity * pot_total - (1 - equity) * state.amount_to_call
        max_val = max(pot_total, state.amount_to_call, 1)
        ev_call = (ev_call_raw / max_val + 1) / 2
        out[idx] = max(0.0, min(1.0, ev_call))
        
        invested = max(0, state.pot_size - state.stack)
        total_initial = state.stack + invested
        out[idx + 1] = invested / max(total_initial, 1)
        
        fold_equity = self._estimate_fold_equity(state)
        normalizer = max(state.pot_size + state.stack, 1)
        gain_if_fold = state.pot_size / normalizer
        gain_if_call = equity * (state.pot_size + state.amount_to_call) / normalizer
        ev_raise = fold_equity * gain_if_fold + (1 - fold_equity) * gain_if_call
        out[idx + 2] = max(0.0, min(1.0, ev_raise))
        
        out[idx + 3] = fold_equity
        out[idx + 4] = self._calculate_implied_odds(state, equity)
        out[idx + 5] = self._calculate_reverse_implied_odds(state, equity)
        
        # Fix: Vrai calcul d'investissement par rapport à notre tapis.
        # Dans game_state, stack = jetons restants. 
        # L'investissement initial max possible sur la main = stack restant actuel + ce qu'on a déjà mis.
        # Approximation: on utilise stack_initial_estimé = 100BB (pour normaliser) car stack varille dynamiquement dans RLCard.
        # Ou plus précis: on regarde notre investissement effectif vs le stack actuel
        inv = max(0, state.pot_size / 2) # Approximation de note investissement si pot partagé
        total_stack_size = state.stack + inv
        commitment = inv / max(total_stack_size, 1)
        out[idx + 6] = min(max(commitment, 0.0), 1.0)
        
        range_adv = self._estimate_range_advantage(state)
        out[idx + 7] = range_adv['nutted']
        out[idx + 8] = range_adv['medium']
        out[idx + 9] = range_adv['weak']
        
        board_cov = self._estimate_board_coverage(state)
        out[idx + 10] = board_cov['high']
        out[idx + 11] = board_cov['medium']
        out[idx + 12] = board_cov['low']
        
        polarization = self._estimate_polarization(state)
        out[idx + 13] = polarization['polarized']
        out[idx + 14] = polarization['merged']
        out[idx + 15] = polarization['capped']
        
        patterns = self._analyze_betting_patterns(state)
        out[idx + 16] = patterns['value']
        out[idx + 17] = patterns['bluff']
        out[idx + 18] = patterns['balanced']
        out[idx + 19] = patterns['exploitative']
        
        # ─── GTO Fundamentals ───────────────────────────
        
        bet_size = state.amount_to_call
        if bet_size > 0:
            mdf = state.pot_size / (state.pot_size + bet_size)
        else:
            mdf = 0.0
        out[idx + 20] = max(0.0, min(1.0, mdf))
        
        effective_bet = state.get_last_aggression_amount()
        if effective_bet <= 0:
            effective_bet = bet_size
        if effective_bet > 0:
            alpha = effective_bet / (state.pot_size + effective_bet)
        else:
            alpha = 0.0
        out[idx + 21] = max(0.0, min(1.0, alpha))
        
        if effective_bet > 0 and state.pot_size > 0:
            raw_ratio = effective_bet / state.pot_size
            bet_to_pot = raw_ratio / (1.0 + raw_ratio)
        else:
            bet_to_pot = 0.0
        out[idx + 22] = max(0.0, min(1.0, bet_to_pot))
        
        streets_remaining = {'preflop': 3, 'flop': 2, 'turn': 1, 'river': 0}
        n_streets = streets_remaining.get(state.street, 0)
        
        if n_streets > 0 and state.pot_size > 0 and state.stack > 0:
            total_ratio = (state.stack + state.pot_size) / state.pot_size
            geo_fraction = total_ratio ** (1.0 / n_streets) - 1.0
            geo_bet = state.pot_size * geo_fraction
            if effective_bet > 0 and geo_bet > 0:
                ratio = effective_bet / geo_bet
                pot_geometry = ratio / (1.0 + ratio)
            else:
                pot_geometry = 0.5
        else:
            pot_geometry = 0.5
        out[idx + 23] = max(0.0, min(1.0, pot_geometry))
        
        # ─── Tier 1 GTO Advanced ────────────────────────
        
        is_ip = 1.0 if state.position in ['BTN', 'CO'] else 0.0
        eq_real_base = 0.95 if is_ip else 0.70
        
        if len(state.board) >= 3:
            flush_made, flush_draw = self._check_flush(state.hole_cards, state.board)
            straight_made, straight_draw = self._check_straight(state.hole_cards, state.board)
            hand_strength = self._evaluate_hand_strength(state.hole_cards, state.board)
            
            if flush_draw > 0 or straight_draw > 0:
                draw_penalty = 0.85
                if flush_draw > 0 and straight_draw > 0:
                    draw_penalty = 0.92
            elif hand_strength > 0.8:
                draw_penalty = 1.05
            elif hand_strength < 0.3:
                draw_penalty = 0.75
            else:
                draw_penalty = 1.0
        else:
            draw_penalty = 1.0
        
        mw_mult = 1.0 - 0.05 * max(0, state.num_active_players - 2)
        equity_realization = eq_real_base * draw_penalty * max(mw_mult, 0.7)
        out[idx + 24] = max(0.0, min(1.0, equity_realization))
        
        blocker_score = 0.0
        if len(state.board) >= 3:
            board_suits = [self._get_suit_char(c) for c in state.board]
            hero_suits = [self._get_suit_char(c) for c in state.hole_cards]
            hero_ranks = [self._card_rank_to_value(c) for c in state.hole_cards]
            board_ranks = [self._card_rank_to_value(c) for c in state.board]
            
            suit_counts = {}
            for s in board_suits:
                suit_counts[s] = suit_counts.get(s, 0) + 1
            dominant_suit = max(suit_counts, key=suit_counts.get)
            dominant_count = suit_counts[dominant_suit]
            
            if dominant_count >= 3:
                for i, hs in enumerate(hero_suits):
                    if hs == dominant_suit and hero_ranks[i] == 14:
                        blocker_score += 0.4
                    elif hs == dominant_suit and hero_ranks[i] >= 12:
                        blocker_score += 0.2
            elif dominant_count >= 2:
                for i, hs in enumerate(hero_suits):
                    if hs == dominant_suit and hero_ranks[i] == 14:
                        blocker_score += 0.2
            
            max_board = max(board_ranks) if board_ranks else 0
            for hr in hero_ranks:
                if hr == max_board and hr >= 12:
                    blocker_score += 0.2
            
            for hr in hero_ranks:
                if hr in board_ranks and hr >= 10:
                    blocker_score += 0.1
            
            board_sorted = sorted(board_ranks)
            for hr in hero_ranks:
                if len(board_sorted) >= 2:
                    for j in range(len(board_sorted) - 1):
                        gap = board_sorted[j+1] - board_sorted[j]
                        if 1 < gap <= 3 and board_sorted[j] < hr < board_sorted[j+1]:
                            blocker_score += 0.1
        
        out[idx + 25] = max(0.0, min(1.0, blocker_score))
        
        protection = 0.0
        if len(state.board) >= 3:
            hand_strength = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex = self._analyze_board_texture(state.board)
            if 0.4 <= hand_strength <= 0.85:
                protection = 1.0 - abs(hand_strength - 0.65) / 0.25
                protection = max(0.0, protection)
                board_danger = board_tex.get('wet', 0) * 0.4 + board_tex.get('coordinated', 0) * 0.3
                protection *= (1.0 + board_danger)
                street_mult = {'flop': 1.0, 'turn': 0.7, 'river': 0.0}
                protection *= street_mult.get(state.street, 0.5)
            elif hand_strength > 0.85:
                protection = 0.1
            else:
                protection = 0.0
        
        out[idx + 26] = max(0.0, min(1.0, protection))
        
        # ─── Tier 2 GTO Advanced ────────────────────────
        nut_adv = 0.5
        if len(state.board) >= 3:
            board_ranks_na = [self._card_rank_to_value(c) for c in state.board]
            max_board_rank = max(board_ranks_na)
            board_tex_na = self._analyze_board_texture(state.board)
            if max_board_rank >= 14:
                nut_adv = 0.70 if state.position in ['BTN', 'CO'] else 0.55
            elif max_board_rank >= 13:
                nut_adv = 0.65 if state.position in ['BTN', 'CO'] else 0.50
            elif max_board_rank <= 8:
                nut_adv = 0.35 if state.position in ['BTN', 'CO'] else 0.60
            else:
                nut_adv = 0.50
            
            coord = board_tex_na.get('coordinated', 0)
            nut_adv -= coord * 0.15
            
            suit_counts_na = {}
            for c in state.board:
                s = self._get_suit_char(c)
                suit_counts_na[s] = suit_counts_na.get(s, 0) + 1
            if max(suit_counts_na.values()) >= 3:
                nut_adv -= 0.1
        out[idx + 27] = max(0.0, min(1.0, nut_adv))
        
        streets_remaining_lev = {'preflop': 3, 'flop': 2, 'turn': 1, 'river': 0}
        n_streets_lev = streets_remaining_lev.get(state.street, 0)
        
        if state.pot_size > 0 and n_streets_lev > 0:
            spr = state.stack / state.pot_size
            spr_norm = min(spr / 10.0, 1.0)
            streets_factor = n_streets_lev / 3.0
            leverage = spr_norm * streets_factor
        else:
            leverage = 0.0
        out[idx + 28] = max(0.0, min(1.0, leverage))
        
        if state.pot_size > 0:
            raw_spr = state.stack / state.pot_size
            eff_spr = raw_spr / (raw_spr + 4.0)
        else:
            eff_spr = 1.0
        out[idx + 29] = max(0.0, min(1.0, eff_spr))
        
        # ─── Tier 3 GTO ──────────────────────────────────
        cr_signal = 0.0
        is_oop = state.position in ['SB', 'BB', 'UTG']
        
        if is_oop and len(state.board) >= 3:
            hand_strength_cr = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex_cr = self._analyze_board_texture(state.board)
            
            if 0.65 <= hand_strength_cr <= 0.90:
                cr_signal = 0.7
                wet = board_tex_cr.get('wet', 0)
                cr_signal *= (1.0 - wet * 0.3)
            elif hand_strength_cr < 0.4 and len(state.board) >= 3:
                flush_m, flush_d = self._check_flush(state.hole_cards, state.board)
                straight_m, straight_d = self._check_straight(state.hole_cards, state.board)
                if flush_d > 0 or straight_d > 0:
                    cr_signal = 0.5
                    if flush_d > 0 and straight_d > 0:
                        cr_signal = 0.65
            if state.amount_to_call <= 0:
                cr_signal *= 0.3
        out[idx + 30] = max(0.0, min(1.0, cr_signal))
        
        eq_denial = 0.0
        if len(state.board) >= 3 and state.street != 'river':
            hand_strength_ed = self._evaluate_hand_strength(state.hole_cards, state.board)
            board_tex_ed = self._analyze_board_texture(state.board)
            
            if 0.45 <= hand_strength_ed <= 0.85:
                base_denial = hand_strength_ed * 0.8
                wet_ed = board_tex_ed.get('wet', 0)
                coord_ed = board_tex_ed.get('coordinated', 0)
                draw_density = wet_ed * 0.5 + coord_ed * 0.3
                
                suit_counts_ed = {}
                for c in state.board:
                    s = self._get_suit_char(c)
                    suit_counts_ed[s] = suit_counts_ed.get(s, 0) + 1
                if 2 in suit_counts_ed.values():
                    draw_density += 0.15
                
                eq_denial = base_denial * (1.0 + draw_density)
                eq_denial *= (1.0 + 0.1 * max(0, state.num_active_players - 2))
        
        out[idx + 31] = max(0.0, min(1.0, eq_denial))
        
        return idx + 32
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
