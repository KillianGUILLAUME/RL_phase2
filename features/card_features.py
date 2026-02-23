from typing import List, Tuple, Dict
import numpy as np
from treys import Card, Evaluator
from core.game_state import GameState
import functools



class UtilsCardFeatures:
    def __init__(self, premium_hands=None, strong_hands=None):
        self.PREMIUM_HANDS = premium_hands or set()
        self.STRONG_HANDS  = strong_hands  or set()
        self.evaluator = Evaluator()


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
            base_str = f"{r1_char}{r2_char}"
        else:
            base_str = f"{r2_char}{r1_char}"
            
        if r1_char == r2_char:
            return base_str
        else:
            return base_str + suited_str
    
    def _normalize_card(self, card: str) -> str:
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

    @functools.lru_cache(maxsize=16384)
    def _cached_evaluate_hand_strength(self, tuple_hole: Tuple[str, ...], tuple_board: Tuple[str, ...]) -> float:
        """
        Évalue la force d'une main avec treys. Modifié pour accepter des tuples (hashable).
        
        Args:
            tuple_hole: tuple de cartes privées ('C5', 'Sa')
            tuple_board: tuple de cartes communes ('D6', 'H9', 'C8')
        
        Returns:
            Score normalisé [0, 1] (1 = nuts)
        """
        if len(tuple_board) < 3:
            return 0.0  # Pas de board = pas d'évaluation

        try:
            # Normalisation des cartes
            hand_norm = [self._normalize_card(c) for c in tuple_hole]
            board_norm = [self._normalize_card(c) for c in tuple_board]
            
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
            print(f"   tuple_hole={tuple_hole}, tuple_board={tuple_board}")
            return 0.5  # Fallback neutre

    def _evaluate_hand_strength(self, hole_cards: List[str], board: List[str]) -> float:
        """Wrapper compatible avec les listes, appelant la version memoisée."""
        return self._cached_evaluate_hand_strength(tuple(hole_cards), tuple(board))

    @functools.lru_cache(maxsize=16384)
    def _cached_estimate_equity(self, tuple_hole: Tuple[str, ...], tuple_board: Tuple[str, ...], num_opponents: int) -> float:
        """
        Estime l'equity (0-1) en combinant Force Actuelle + Potentiel (Tirages). (Cache Version)
        """
        if len(tuple_hole) < 2:
            return 0.0
        
        # --- CAS 1 : PREFLOP (Pas de changement majeur) ---
        if not tuple_board:
            hand_str = self._get_preflop_hand_string(list(tuple_hole))
            if hand_str in self.PREMIUM_HANDS: base = 0.80
            elif hand_str in self.STRONG_HANDS: base = 0.60
            elif hand_str.endswith('s'): base = 0.50 # Bonus pour suited
            elif 'A' in hand_str or 'K' in hand_str: base = 0.45
            else: base = 0.35
            
            # Ajustement nb joueurs (plus on est nombreux, moins on a d'equity brute)
            return base * (0.95 ** max(0, num_opponents - 1))

        # --- CAS 2 : POSTFLOP (La grosse correction) ---
        
        # A. La force actuelle (calculée avec Treys ou la méthode corrigée)
        current_strength = self._cached_evaluate_hand_strength(tuple_hole, tuple_board)
        
        # B. Le potentiel (Draws)
        draw_bonus = 0.0
        
        hole_list = list(tuple_hole)
        board_list = list(tuple_board)
        
        # On ne calcule les bonus que si la main n'est pas déjà "Faite" (Brelan ou mieux)
        if current_strength < 0.7:
            # Check Flush Draw
            flush_made, flush_draw = self._check_flush(hole_list, board_list)
            if flush_draw > 0: 
                draw_bonus += 0.20  # ~20% d'equity pour un tirage couleur
            
            # Check Straight Draw
            straight_made, straight_draw = self._check_straight(hole_list, board_list)
            if straight_draw > 0:
                draw_bonus += 0.10  # ~10% pour un tirage quinte
                
            # Check Overcards (Si j'ai AK sur un board 2-5-9)
            overcards = self._count_overcards(hole_list, board_list)
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

    def _estimate_equity(self, hole_cards: List[str], board: List[str], num_opponents: int) -> float:
        """Wrapper compatible avec les listes, appelant la version memoisée."""
        return self._cached_estimate_equity(tuple(hole_cards), tuple(board), num_opponents)
        
    @functools.lru_cache(maxsize=16384)
    def _cached_monte_carlo_equity(self, tuple_hole: Tuple[str, ...], tuple_board: Tuple[str, ...], num_opponents: int, num_simulations: int = 200) -> float:
        """
        Calcule une équité plus précise via simulations de Monte Carlo aléatoires (Fast).
        """
        if len(tuple_hole) < 2: return 0.0
        
        hand_norm = [self._normalize_card(c) for c in tuple_hole]
        board_norm = [self._normalize_card(c) for c in tuple_board]
        my_ints = [Card.new(c) for c in hand_norm]
        board_ints = [Card.new(c) for c in board_norm]
        
        known_cards = set(my_ints + board_ints)
        deck = [Card.new(r + s) for r in '23456789TJQKA' for s in 'shdc']
        remaining = [c for c in deck if c not in known_cards]
        
        wins = 0
        ties = 0
        
        cards_for_board = 5 - len(board_ints)
        cards_for_opps = 2 * num_opponents
        total_needed = cards_for_board + cards_for_opps
        
        if len(remaining) < total_needed:
            return 0.5
            
        evaluator = Evaluator()
        
        for _ in range(num_simulations):
            np.random.shuffle(remaining)
            dealt = remaining[:total_needed]
            
            sim_board = board_ints + dealt[:cards_for_board]
            opp_cards = dealt[cards_for_board:]
            
            my_score = evaluator.evaluate(sim_board, my_ints)
            best_opp_score = 7463 # pire score possible (1 est royal flush)
            
            for i in range(num_opponents):
                opp_hand = [opp_cards[i*2], opp_cards[i*2 + 1]]
                opp_score = evaluator.evaluate(sim_board, opp_hand)
                if opp_score < best_opp_score:
                    best_opp_score = opp_score
                    
            if my_score < best_opp_score:
                wins += 1
            elif my_score == best_opp_score:
                ties += 1
                
        # Chaque tie est divisé par eq (approximation simplifiée, on l'estime à la part d'opponents impliqués)
        return (wins + (ties / (num_opponents + 1))) / num_simulations

    def _monte_carlo_equity(self, hole_cards: List[str], board: List[str], num_opponents: int, num_simulations: int = 200) -> float:
        """Wrapper de liste pour Monte Carlo."""
        return self._cached_monte_carlo_equity(tuple(hole_cards), tuple(board), num_opponents, num_simulations)
    
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




class CardFeatures(UtilsCardFeatures):
    def __init__(self, premium_hands, strong_hands, evaluator):
        super().__init__(premium_hands, strong_hands)
        self.evaluator           = evaluator

    
    
    def _extract_card_features(self, state: GameState, out: np.ndarray, idx: int) -> int:
        hole_cards = state.hole_cards
        board = state.board
        
        # Encodage des cartes (2)
        if len(hole_cards) >= 2:
            rank1 = self._card_rank_to_value(hole_cards[0])
            rank2 = self._card_rank_to_value(hole_cards[1])
            out[idx] = rank1 / 14.0
            out[idx + 1] = rank2 / 14.0
        else:
            out[idx] = 0.0
            out[idx + 1] = 0.0
        
        # Suited (1)
        if len(hole_cards) >= 2:
            s1 = self._get_suit_char(hole_cards[0])
            s2 = self._get_suit_char(hole_cards[1])
            suited = 1.0 if s1 == s2 else 0.0
        else:
            suited = 0.0
        out[idx + 2] = suited
        
        # Pocket pair (1)
        if len(hole_cards) >= 2:
            r1 = self._get_rank_char(hole_cards[0])
            r2 = self._get_rank_char(hole_cards[1])
            pocket_pair = 1.0 if r1 == r2 else 0.0
        else:
            pocket_pair = 0.0
        out[idx + 3] = pocket_pair
        
        # Hand strength preflop (2)
        hand_str = self._get_preflop_hand_string(hole_cards)
        is_premium = 1.0 if hand_str in self.PREMIUM_HANDS else 0.0
        is_strong = 1.0 if hand_str in self.STRONG_HANDS else 0.0
        out[idx + 4] = is_premium
        out[idx + 5] = is_strong
        
        # Hand strength postflop (1)
        if state.street != 'preflop' and len(board) >= 3:
            postflop_strength = self._evaluate_hand_strength(hole_cards, board)
        else:
            postflop_strength = 0.0
        out[idx + 6] = postflop_strength
        
        # Equity estimée (1)
        equity = self._estimate_equity(hole_cards, board, state.num_active_players)
        out[idx + 7] = equity
        
        # Flush draws (2)
        flush_made, flush_draw = self._check_flush(hole_cards, board)
        out[idx + 8] = flush_made
        out[idx + 9] = flush_draw
        
        # Straight draws (2)
        straight_made, straight_draw = self._check_straight(hole_cards, board)
        out[idx + 10] = straight_made
        out[idx + 11] = straight_draw
        
        # Overcards (1)
        overcards = self._count_overcards(hole_cards, board)
        out[idx + 12] = overcards / 2.0  # Normalisation (max 2)
        
        # Hand made (4): pair, two_pair, trips, quads
        pair, two_pair, trips, quads = self._check_hand_made(hole_cards, board)
        out[idx + 13] = pair
        out[idx + 14] = two_pair
        out[idx + 15] = trips
        out[idx + 16] = quads
        
        # Board texture (5)
        if len(board) >= 3:
            texture = self._analyze_board_texture(board)
            out[idx + 17] = texture['coordinated']
            out[idx + 18] = texture['wet']
            out[idx + 19] = texture['paired']
            out[idx + 20] = texture['high_cards']
            out[idx + 21] = texture['monotone']
        else:
            for i in range(5):
                out[idx + 17 + i] = 0.0
        
        return idx + 22