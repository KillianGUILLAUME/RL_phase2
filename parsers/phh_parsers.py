"""
Parser pour les fichiers .phh (Pluribus Hand History).
Extrait toutes les décisions (état, action) pour l'entraînement.
"""

import zipfile
import io
import re
from typing import List, Dict, Tuple
from collections import defaultdict

class PHHParser:
    """
    Parser pour fichiers .phh.
    """
    
    # Mapping des actions
    ACTION_MAP = {
        'f': 'fold',
        'cc': 'check_call',  # Check si possible, sinon call
        'cbr': 'bet_raise',  # Bet si première action, sinon raise
    }
    
    def __init__(self, zip_path: str = "data/poker-hand-histories.zip"):
        self.zip_path = zip_path
        self.hands_data = []
    
    def parse_all(self, max_hands: int = None) -> List[Dict]:
        """
        Parser tous les fichiers .phh du zip.
        
        Args:
            max_hands: Limite le nombre de mains (None = toutes)
        
        Returns:
            Liste de dictionnaires avec les données de chaque main
        """
        print(f"📂 Ouverture de {self.zip_path}")
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            phh_files = [f for f in zf.namelist() if f.endswith('.phh')]
            print(f"✅ {len(phh_files)} fichiers .phh trouvés")
            
            for i, phh_file in enumerate(phh_files):
                if max_hands and i >= max_hands:
                    break
                
                with zf.open(phh_file, 'r') as f_bin:
                    content = io.TextIOWrapper(f_bin, encoding='utf-8').read()
                    hand_data = self._parse_hand(content)
                    if hand_data:
                        self.hands_data.append(hand_data)
                
                if (i + 1) % 1000 == 0:
                    print(f"   ... {i + 1} mains parsées")
        
        print(f"✅ Total : {len(self.hands_data)} mains parsées avec succès\n")
        return self.hands_data
    
    def _parse_hand(self, content: str) -> Dict:
        """
        Parser une main complète depuis le contenu du fichier.
        """
        try:
            # Extraire les variables Python-like
            data = {}
            exec(content, {}, data)
            
            # Vérifier que les clés essentielles existent
            required_keys = ['actions', 'blinds_or_straddles', 'starting_stacks', 
                           'finishing_stacks', 'players']
            if not all(k in data for k in required_keys):
                return None
            
            # Parser les actions
            decisions = self._extract_decisions(data)
            
            return {
                'raw_data': data,
                'decisions': decisions,
                'num_players': len(data['players']),
                'players': data['players']
            }
        
        except Exception as e:
            # print(f"⚠️  Erreur parsing: {e}")
            return None
    
    def _extract_decisions(self, data: Dict) -> List[Dict]:
        """
        Extraire toutes les décisions (état, action) d'une main.
        
        Returns:
            Liste de {
                'player_id': int,
                'hole_cards': [str, str],
                'board': [str, ...],
                'action_type': str,
                'amount': int or None,
                'pot_size': int,
                'stack': int,
                'position': str,
                'num_active_players': int
            }
        """
        decisions = []
        
        # État de la main
        n_players = len(data['players'])
        stacks = list(data['starting_stacks'])
        pot = sum(data['blinds_or_straddles'])
        hole_cards = [None] * n_players
        board = []
        active_players = set(range(n_players))
        
        # Identifier les positions
        sb_pos = 0  # Small blind
        bb_pos = 1  # Big blind
        btn_pos = (n_players - 1) if n_players > 2 else 0  # Button
        
        # Parser chaque action
        for action_str in data['actions']:
            parts = action_str.split()
            action_type = parts[0]
            
            # Deal hole cards
            if action_type == 'd' and parts[1] == 'dh':
                player_id = int(parts[2][1:]) - 1  # 'p1' -> 0
                cards = [parts[3][:2], parts[3][2:]]  # 'TcQc' -> ['Tc', 'Qc']
                hole_cards[player_id] = cards
            
            # Deal board
            elif action_type == 'd' and parts[1] == 'db':
                new_cards = parts[2]
                # Flop = 6 chars, Turn/River = 2 chars
                if len(new_cards) == 6:
                    board.extend([new_cards[i:i+2] for i in range(0, 6, 2)])
                else:
                    board.append(new_cards)
            
            # Action d'un joueur
            elif action_type.startswith('p'):
                player_id = int(action_type[1:]) - 1
                
                # Ne traiter que les joueurs actifs
                if player_id not in active_players:
                    continue
                
                decision_type = parts[1]
                amount = None
                
                # Fold
                if decision_type == 'f':
                    active_players.remove(player_id)
                    action = 'fold'
                
                # Check/Call
                elif decision_type == 'cc':
                    action = 'check_call'
                    # TODO: Calculer le montant du call si nécessaire
                
                # Bet/Raise
                elif decision_type == 'cbr':
                    action = 'bet_raise'
                    amount = int(parts[2])
                    stacks[player_id] -= amount
                    pot += amount
                
                else:
                    continue
                
                # Déterminer la position
                if player_id == btn_pos:
                    position = 'BTN'
                elif player_id == sb_pos:
                    position = 'SB'
                elif player_id == bb_pos:
                    position = 'BB'
                else:
                    # Position relative au bouton
                    dist = (player_id - btn_pos) % n_players
                    if dist <= n_players // 2:
                        position = 'EP'  # Early position
                    else:
                        position = 'MP'  # Middle position
                
                # Enregistrer la décision
                decisions.append({
                    'player_id': player_id,
                    'player_name': data['players'][player_id],
                    'hole_cards': hole_cards[player_id],
                    'board': board.copy(),
                    'street': self._get_street(board),
                    'action': action,
                    'amount': amount,
                    'pot_size': pot,
                    'stack': stacks[player_id],
                    'position': position,
                    'num_active_players': len(active_players),
                    'is_pluribus': data['players'][player_id] == 'Pluribus'
                })
        
        return decisions
    
    @staticmethod
    def _get_street(board: List[str]) -> str:
        """Déterminer la street depuis le board."""
        if len(board) == 0:
            return 'preflop'
        elif len(board) == 3:
            return 'flop'
        elif len(board) == 4:
            return 'turn'
        elif len(board) == 5:
            return 'river'
        return 'unknown'


# ============================================================================
# FONCTION DE TEST
# ============================================================================

def test_parser():
    """Test rapide du parser."""
    parser = PHHParser("data/poker-hand-histories.zip")
    hands = parser.parse_all(max_hands=10)
    
    print("=" * 70)
    print("🔍 ANALYSE DES 10 PREMIÈRES MAINS")
    print("=" * 70)
    
    total_decisions = 0
    pluribus_decisions = 0
    
    for i, hand in enumerate(hands):
        decisions = hand['decisions']
        plur_dec = sum(1 for d in decisions if d['is_pluribus'])
        
        total_decisions += len(decisions)
        pluribus_decisions += plur_dec
        
        print(f"\nMain #{i+1}:")
        print(f"  Joueurs: {', '.join(hand['players'])}")
        print(f"  Décisions totales: {len(decisions)}")
        print(f"  Décisions Pluribus: {plur_dec}")
        
        # Afficher les 3 premières décisions
        print(f"  Premières actions:")
        for dec in decisions[:3]:
            print(f"    - {dec['player_name']} ({dec['position']}, {dec['street']}): "
                  f"{dec['action']} {dec['amount'] if dec['amount'] else ''}")
    
    print("\n" + "=" * 70)
    print(f"📊 STATISTIQUES GLOBALES")
    print("=" * 70)
    print(f"Total décisions: {total_decisions}")
    print(f"Décisions Pluribus: {pluribus_decisions} "
          f"({100*pluribus_decisions/total_decisions:.1f}%)")


if __name__ == "__main__":
    test_parser()
