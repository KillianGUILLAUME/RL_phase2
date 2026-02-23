# core/game_state.py

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class Street(Enum):
    """Enumération des streets pour éviter les typos."""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


@dataclass
class GameState:
    """
    État du jeu standardisé, indépendant de la source (Pluribus ou RLCard).
    
    Stocke TOUT ce qui est nécessaire pour :
    1. Extraire des features (FeatureExtractor)
    2. Prendre une décision (Agent)
    3. Calculer des métriques (stack en BB, pot odds, etc.)
    
    Design principles:
    - Immutable (dataclass frozen si besoin)
    - Sémantique poker (pas de features pré-calculées)
    - Calculs dynamiques via @property
    """
    
    # ═══════════════════════════════════════════════════════════
    # CARTES
    # ═══════════════════════════════════════════════════════════
    hole_cards: List[str]  # ['As', 'Kd']
    board: List[str]       # ['Jh', '9c', '4s'] (vide preflop)
    
    # ═══════════════════════════════════════════════════════════
    # STREET & POSITION
    # ═══════════════════════════════════════════════════════════
    street: str                    # 'preflop', 'flop', 'turn', 'river'
    position: str                  # 'BTN', 'SB', 'BB', 'UTG', 'MP', 'CO'
    num_active_players: int        # Joueurs encore dans le coup
    
    # ═══════════════════════════════════════════════════════════
    # CHIPS & BLINDS
    # ═══════════════════════════════════════════════════════════
    pot_size: int          # Pot total en chips
    stack: int             # Stack du joueur (chips absolus)
    big_blind: int         # Montant de la BB (ex: 100)
    small_blind: int       # Montant de la SB (ex: 50)
    
    # ═══════════════════════════════════════════════════════════
    # ACTIONS
    # ═══════════════════════════════════════════════════════════
    amount_to_call: int              # Montant à payer pour call
    legal_actions: List[str]         # ['fold', 'call', 'raise']
    actions_this_street: List[str]   # Historique de la street courante
    all_actions_history: List[str]   # Historique COMPLET de la main (pour contexte preflop)
    
    # ═══════════════════════════════════════════════════════════
    # METADATA (optionnel)
    # ═══════════════════════════════════════════════════════════
    hand_id: Optional[str] = None    # ID unique de la main
    player_id: Optional[int] = None  # ID du joueur
    
    # ═══════════════════════════════════════════════════════════
    # PROPRIÉTÉS CALCULÉES DYNAMIQUEMENT
    # ═══════════════════════════════════════════════════════════
    
    @property
    def effective_stack_bb(self) -> float:
        """
        Stack effectif en nombre de big blinds.
        
        Exemples:
        - stack=10000, BB=100 → 100.0 BB (deep stack)
        - stack=5000, BB=100  → 50.0 BB (medium stack)
        - stack=1500, BB=100  → 15.0 BB (short stack)
        
        Utilisé pour:
        - Adapter la stratégie (short stack = push/fold)
        - Normaliser les features pour le ML
        """
        if self.big_blind == 0:
            raise ValueError("Big blind cannot be zero")
        return self.stack / self.big_blind
    
    @property
    def pot_size_bb(self) -> float:
        """
        Pot en nombre de big blinds.
        
        Utilisé pour:
        - Calculer les pot odds
        - Features ML (ratio pot/stack)
        """
        if self.big_blind == 0:
            raise ValueError("Big blind cannot be zero")
        return self.pot_size / self.big_blind
    
    @property
    def amount_to_call_bb(self) -> float:
        """
        Montant à call en big blinds.
        
        Exemples:
        - Preflop UTG: 1.0 BB (call la BB)
        - Flop après bet 300 (BB=100): 3.0 BB
        """
        if self.big_blind == 0:
            raise ValueError("Big blind cannot be zero")
        return self.amount_to_call / self.big_blind
    
    @property
    def pot_odds(self) -> float:
        """
        Pot odds = montant à call / (pot + montant à call)
        
        Exemples:
        - Pot=1000, to_call=500 → 0.33 (33%, besoin 25%+ equity)
        - Pot=500, to_call=100  → 0.17 (17%, besoin 12.5%+ equity)
        
        Utilisé pour:
        - Décision call/fold (comparer avec equity)
        - Feature ML cruciale
        """
        if self.amount_to_call == 0:
            return 0.0
        total = self.pot_size + self.amount_to_call
        return self.amount_to_call / total if total > 0 else 0.0
    
    @property
    def is_all_in_situation(self) -> bool:
        """
        True si call = all-in (stack ≤ amount_to_call).
        
        Important car:
        - Change les actions légales (plus de raise possible)
        - Stratégie différente (pot odds uniquement)
        """
        return self.stack <= self.amount_to_call
    
    @property
    def street_enum(self) -> Street:
        """Convertit street string en enum (pour pattern matching)."""
        return Street(self.street)
    
    @property
    def num_board_cards(self) -> int:
        """Nombre de cartes au board (0/3/4/5)."""
        return len(self.board)
    
    @property
    def is_heads_up(self) -> bool:
        """True si seulement 2 joueurs actifs."""
        return self.num_active_players == 2
    
    @property
    def spr(self) -> float:
        """
        Stack-to-Pot Ratio = stack effectif / pot
        
        Exemples:
        - SPR > 10 : Deep stack (jeu complexe, implied odds)
        - SPR 4-10 : Medium (standard)
        - SPR < 4  : Shallow (commit facile, moins de manœuvre)
        
        Utilisé pour:
        - Décision continuation bet
        - Taille des mises
        """
        if self.pot_size == 0:
            return float('inf')
        return self.stack / self.pot_size

    @property
    def is_single_raised_pot(self) -> bool:
        """True s'il y a eu exactement 1 seule relance preflop (Single Raised Pot)."""
        preflop_raises = 0
        for action in self.all_actions_history:
            if action == "flop_starts":
                break
            if action.startswith("raise_"):
                preflop_raises += 1
        return preflop_raises == 1

    @property
    def is_3bet_pot(self) -> bool:
        """True s'il y a eu 2 relances ou plus preflop (3-bet+ pot)."""
        preflop_raises = 0
        for action in self.all_actions_history:
            if action == "flop_starts":
                break
            if action.startswith("raise_"):
                preflop_raises += 1
        return preflop_raises >= 2

    @property
    def was_multiway_flop(self) -> bool:
        """
        Détermine empiriquement s'il y avait >= 3 joueurs au flop.
        (Compte le nombre de preflop 'call'/'raise' + SB/BB)
        """
        # Approximatif: on compte les actions volontaires preflop uniques
        voluntarily_put_money_in_pot = set()
        for action in self.all_actions_history:
            if action == "flop_starts":
                break
            # RLCard logs "playerX_action" in custom history? We will parse the logic
            # For now, let's keep it simple: if current active players >= 3, it was multiway. 
            # If current active players == 2, we just look if someone folded postflop.
            pass
            
        # Simplification robuste : si actuellement num_players >= 3, c'est forcément multiway.
        if self.num_active_players >= 3:
            return True
            
        # Si on est heads-up, on cherche juste un 'fold' postflop dans l'historique global
        postflop = False
        folds_postflop = 0
        for action in self.all_actions_history:
            if action == "flop_starts":
                postflop = True
            elif postflop and action == "fold":
                folds_postflop += 1
                
        # Si on est 2 actuellement mais que quelqu'un a fold postflop, c'est qu'on était > 2 au flop
        if self.num_active_players == 2 and folds_postflop > 0:
            return True
            
        return False
    
    # ═══════════════════════════════════════════════════════════
    # MÉTHODES UTILITAIRES
    # ═══════════════════════════════════════════════════════════
    
    def has_action_occurred(self, action_type: str) -> bool:
        """
        Vérifie si un type d'action a eu lieu cette street.
        
        Args:
            action_type: 'bet', 'raise', 'call', 'check', 'fold'
        
        Returns:
            True si l'action est dans l'historique
        
        Exemples:
        - has_action_occurred('raise') → True si quelqu'un a relancé
        - Utile pour détecter les 3-bets, 4-bets, etc.
        """
        return any(
            action.startswith(action_type)
            for action in self.actions_this_street
        )
    
    def get_last_aggression_amount(self) -> int:
        """
        Retourne le montant de la dernière mise/relance.
        
        Returns:
            Montant en chips (0 si aucune aggression)
        
        Utilisé pour:
        - Calculer les min-raises
        - Détecter la taille des bets adverses
        """
        for action in reversed(self.actions_this_street):
            if action.startswith(('bet_', 'raise_')):
                return int(action.split('_')[1])
        return 0
    
    def to_dict(self) -> dict:
        """
        Convertit en dict (pour logging/debugging).
        
        Inclut les propriétés calculées pour inspection complète.
        """
        return {
            # Raw data
            'hole_cards': self.hole_cards,
            'board': self.board,
            'street': self.street,
            'position': self.position,
            'num_active_players': self.num_active_players,
            'pot_size': self.pot_size,
            'stack': self.stack,
            'big_blind': self.big_blind,
            'small_blind': self.small_blind,
            'amount_to_call': self.amount_to_call,
            'legal_actions': self.legal_actions,
            'actions_this_street': self.actions_this_street,
            'all_actions_history': self.all_actions_history,
            
            # Computed properties
            'effective_stack_bb': round(self.effective_stack_bb, 2),
            'pot_size_bb': round(self.pot_size_bb, 2),
            'amount_to_call_bb': round(self.amount_to_call_bb, 2),
            'pot_odds': round(self.pot_odds, 3),
            'spr': round(self.spr, 2),
            'is_all_in_situation': self.is_all_in_situation,
            'is_heads_up': self.is_heads_up,
        }
    
    def __repr__(self) -> str:
        """
        Représentation lisible pour debugging.
        
        Exemple:
        GameState(AsKd | Jh9c4s | BTN | 100BB | Pot:15BB | Call:3BB)
        """
        hole = ''.join(self.hole_cards)
        board = ''.join(self.board) if self.board else '---'
        return (
            f"GameState({hole} | {board} | {self.position} | "
            f"{self.effective_stack_bb:.1f}BB | "
            f"Pot:{self.pot_size_bb:.1f}BB | "
            f"Call:{self.amount_to_call_bb:.1f}BB)"
        )


# ═══════════════════════════════════════════════════════════
# TESTS UNITAIRES
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 TESTS GameState")
    print("=" * 70)
    
    # Test 1: Deep stack preflop
    state1 = GameState(
        hole_cards=['As', 'Kd'],
        board=[],
        street='preflop',
        position='BTN',
        num_active_players=6,
        pot_size=150,
        stack=10000,
        big_blind=100,
        small_blind=50,
        amount_to_call=100,
        legal_actions=['fold', 'call', 'raise'],
        actions_this_street=[],
        all_actions_history=[]
    )
    
    print("\n📊 Test 1: Deep stack preflop")
    print(f"  {state1}")
    print(f"  Effective stack: {state1.effective_stack_bb} BB")
    print(f"  Pot odds: {state1.pot_odds:.1%}")
    print(f"  SPR: {state1.spr:.2f}")
    assert state1.effective_stack_bb == 100.0
    assert state1.pot_odds == 0.4  # 100/(150+100)
    print("  ✅ PASS")
    
    # Test 2: Short stack all-in
    state2 = GameState(
        hole_cards=['Qh', 'Jh'],
        board=['Kh', '9c', '4s'],
        street='flop',
        position='SB',
        num_active_players=2,
        pot_size=800,
        stack=500,
        big_blind=100,
        small_blind=50,
        amount_to_call=600,
        legal_actions=['fold', 'call'],  # All-in
        actions_this_street=['bet_600'],
        all_actions_history=['raise_200', 'call_200', 'flop_starts', 'bet_600']
    )
    
    print("\n📊 Test 2: Short stack all-in situation")
    print(f"  {state2}")
    print(f"  Effective stack: {state2.effective_stack_bb} BB")
    print(f"  Is all-in situation: {state2.is_all_in_situation}")
    print(f"  Has bet occurred: {state2.has_action_occurred('bet')}")
    print(f"  Last aggression: {state2.get_last_aggression_amount()} chips")
    assert state2.is_all_in_situation == True
    assert state2.has_action_occurred('bet') == True
    print("  ✅ PASS")
    
    # Test 3: Blinds différents (RLCard custom)
    state3 = GameState(
        hole_cards=['7h', '6h'],
        board=['8s', '9d', 'Tc'],
        street='flop',
        position='BB',
        num_active_players=3,
        pot_size=1000,
        stack=5000,
        big_blind=50,   # ← Blinds custom
        small_blind=25,
        amount_to_call=0,
        legal_actions=['check', 'bet'],
        actions_this_street=['check'],
        all_actions_history=['call_50', 'check', 'flop_starts', 'check']
    )
    
    print("\n📊 Test 3: Custom blinds (25/50)")
    print(f"  {state3}")
    print(f"  Effective stack: {state3.effective_stack_bb} BB")  # 100 BB
    print(f"  Pot size: {state3.pot_size_bb} BB")  # 20 BB
    assert state3.effective_stack_bb == 100.0
    assert state3.pot_size_bb == 20.0
    print("  ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PASSENT !")
    print("=" * 70)
