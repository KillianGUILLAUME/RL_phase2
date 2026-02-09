from core.game_state import GameState
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PluribusAdapter:
    """
    Convertit une décision Pluribus (.phh) en GameState standardisé.
    
    Gère :
    - Extraction des blinds (réelles ou par défaut)
    - Calcul de amount_to_call depuis l'historique d'actions
    - Détermination des actions légales
    - Reconstruction de l'historique de la street
    """
    
    # Blinds par défaut (Pluribus standard)
    DEFAULT_SMALL_BLIND = 50
    DEFAULT_BIG_BLIND = 100
    
    @staticmethod
    def to_game_state(decision: Dict) -> GameState:
        """
        Convertit un dict de décision Pluribus en GameState.
        
        Args:
            decision: Dict extrait par PHHParser avec les clés:
                - hole_cards: List[str] (ex: ['As', 'Kd'])
                - board: List[str] (ex: ['Jh', '9c', '4s'])
                - street: str ('preflop', 'flop', 'turn', 'river')
                - position: str ('BTN', 'SB', 'BB', 'UTG', etc.)
                - pot_size: int (en chips)
                - stack: int (en chips)
                - num_active_players: int
                - action_history: List[Dict] (optionnel, avec actions précédentes)
                - big_blind: int (optionnel)
                - small_blind: int (optionnel)
        
        Returns:
            GameState standardisé
        """
        
        # === 1. Extraction des blinds ===
        big_blind = decision.get('big_blind', PluribusAdapter.DEFAULT_BIG_BLIND)
        small_blind = decision.get('small_blind', PluribusAdapter.DEFAULT_SMALL_BLIND)
        
        # === 2. Extraction de l'historique d'actions ===
        action_history = decision.get('action_history', [])
        actions_this_street = PluribusAdapter._extract_street_actions(
            action_history, 
            decision['street']
        )

        position = PluribusAdapter.infer_player_position(decision['row'])
        
        # === 3. Calcul de amount_to_call ===
        amount_to_call = PluribusAdapter._calculate_amount_to_call(
            actions_this_street,
            position,
            decision['street'],
            big_blind,
            small_blind
        )
        
        # === 4. Détermination des actions légales ===
        legal_actions = PluribusAdapter._get_legal_actions(
            amount_to_call,
            decision['stack']
        )
        
        # === 5. Construction du GameState ===
        return GameState(
            hole_cards=decision['hole_cards'],
            board=decision.get('board', []),
            street=decision['street'],
            position=decision['position'],
            num_active_players=decision['num_active_players'],
            pot_size=decision['pot_size'],
            stack=decision['stack'],
            big_blind=big_blind,
            small_blind=small_blind,
            amount_to_call=amount_to_call,
            legal_actions=legal_actions,
            actions_this_street=actions_this_street
        )
    
    @staticmethod
    def _extract_street_actions(
        action_history: List[Dict], 
        current_street: str
    ) -> List[str]:
        """
        Extrait les actions de la street courante depuis l'historique complet.
        
        Args:
            action_history: Liste de dicts avec 'street', 'action', 'amount'
            current_street: 'preflop', 'flop', 'turn', 'river'
        
        Returns:
            Liste d'actions sur cette street (ex: ['call', 'raise_200', 'fold'])
        """
        actions = []
        for entry in action_history:
            if entry.get('street') != current_street:
                continue
            
            action = entry['action']
            amount = entry.get('amount', 0)
            
            # Formater l'action
            if action in ['fold', 'check', 'call']:
                actions.append(action)
            elif action in ['bet', 'raise']:
                actions.append(f"{action}_{amount}")
            else:
                logger.warning(f"Action inconnue : {action}")
        
        return actions
    
    @staticmethod
    def infer_player_position(row):
        """
        Déduire la position du joueur décisionnaire
        
        Logique: Le joueur décisionnaire est celui dont le stack
                correspond à stack_before
        """
        stack_before = row['stack_before']
        
        for pos in ['SB', 'BB', 'UTG', 'MP', 'CO', 'BTN']:
            col_stack = f'pos_{pos}_stack'
            col_in_hand = f'pos_{pos}_in_hand'
            
            if col_stack in row and col_in_hand in row:
                if row[col_in_hand] == 1 and row[col_stack] == stack_before:
                    return pos
        
        # Fallback: si pas trouvé, retourner BTN
        return 'BTN'
    
    @staticmethod
    def _calculate_amount_to_call(
        actions_this_street: List[str],
        position: str,
        street: str,
        big_blind: int,
        small_blind: int
    ) -> int:
        """
        Calcule le montant à payer pour call.
        
        Logique :
        - Preflop : Dépend de la position (BB déjà investi, SB à compléter)
        - Postflop : Dépend de la dernière mise/relance
        
        Args:
            actions_this_street: Actions déjà prises sur cette street
            position: Position du joueur
            street: Street courante
            big_blind: Montant de la BB
            small_blind: Montant de la SB
        
        Returns:
            Montant à call (0 si check possible)
        """
        
        # === Cas 1 : Preflop ===
        if street == 'preflop':
            # Si personne n'a relancé, c'est le montant de la BB
            current_bet = big_blind
            
            # Parcourir les actions pour trouver la dernière mise
            for action in actions_this_street:
                if action.startswith('raise_'):
                    current_bet = int(action.split('_')[1])
                elif action.startswith('bet_'):
                    current_bet = int(action.split('_')[1])
            
            # Ajustement selon la position
            if position == 'BB':
                # BB a déjà investi big_blind
                return max(0, current_bet - big_blind)
            elif position == 'SB':
                # SB a déjà investi small_blind
                return max(0, current_bet - small_blind)
            else:
                # Autres positions n'ont rien investi
                return current_bet
        
        # === Cas 2 : Postflop ===
        else:
            current_bet = 0
            
            # Trouver la dernière mise/relance
            for action in actions_this_street:
                if action.startswith(('bet_', 'raise_')):
                    current_bet = int(action.split('_')[1])
            
            return current_bet
    
    @staticmethod
    def _get_legal_actions(amount_to_call: int, stack: int) -> List[str]:
        """
        Détermine les actions légales selon le contexte.
        
        Args:
            amount_to_call: Montant à payer pour call
            stack: Stack du joueur
        
        Returns:
            Liste d'actions légales (ex: ['fold', 'call', 'raise'])
        """
        
        # Si amount_to_call == 0 → check/bet disponibles
        if amount_to_call == 0:
            actions = ['check', 'bet']
        else:
            actions = ['fold', 'call']
            
            # Raise possible seulement si on a assez de jetons
            # (au minimum call + 1 BB de relance)
            if stack > amount_to_call:
                actions.append('raise')
        
        return actions


# === Tests unitaires ===
if __name__ == '__main__':
    # Test 1 : Preflop UTG face à BB
    decision1 = {
        'hole_cards': ['As', 'Kd'],
        'board': [],
        'street': 'preflop',
        'position': 'UTG',
        'pot_size': 150,
        'stack': 10000,
        'num_active_players': 6,
        'big_blind': 100,
        'small_blind': 50,
        'action_history': []
    }
    
    state1 = PluribusAdapter.to_game_state(decision1)
    print("Test 1 - Preflop UTG:")
    print(f"  Amount to call: {state1.amount_to_call}")  # Devrait être 100
    print(f"  Legal actions: {state1.legal_actions}")    # ['fold', 'call', 'raise']
    print(f"  Stack en BB: {state1.effective_stack_bb}") # 100.0
    
    # Test 2 : Flop après un bet
    decision2 = {
        'hole_cards': ['Qh', 'Jh'],
        'board': ['Kh', '9c', '4s'],
        'street': 'flop',
        'position': 'BTN',
        'pot_size': 500,
        'stack': 8500,
        'num_active_players': 3,
        'big_blind': 100,
        'small_blind': 50,
        'action_history': [
            {'street': 'flop', 'action': 'bet', 'amount': 300}
        ]
    }
    
    state2 = PluribusAdapter.to_game_state(decision2)
    print("\nTest 2 - Flop BTN face à bet:")
    print(f"  Amount to call: {state2.amount_to_call}")  # 300
    print(f"  Legal actions: {state2.legal_actions}")    # ['fold', 'call', 'raise']
    print(f"  Actions this street: {state2.actions_this_street}")  # ['bet_300']
