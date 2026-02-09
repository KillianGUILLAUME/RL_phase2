from core.game_state import GameState
from typing import Dict, List, Optional, Any
import logging
import pprint

logger = logging.getLogger(__name__)


class RLCardAdapter:
    """
    Convertit les observations RLCard en GameState standardisé.
    
    RLCard fournit des observations sous forme de dict avec:
    - obs['raw_obs']: État brut du jeu
    - obs['action_record']: Historique des actions
    - obs['legal_actions']: Actions possibles (indices numériques)
    
    Référence: https://github.com/datamllab/rlcard/blob/master/rlcard/envs/nolimitholdem.py
    """
    
    # Mapping des actions RLCard (indices) vers noms
    ACTION_MAPPING = {
        0: 'call',
        1: 'raise',
        2: 'fold',
        3: 'check',
        4: 'all_in'  # RLCard peut avoir un all-in explicite
    }
    
    # Mapping positions RLCard (indices) vers noms standards
    POSITION_MAPPING = {
        0: 'SB',   # Small Blind
        1: 'BB',   # Big Blind
        2: 'UTG',  # Under The Gun
        3: 'MP',   # Middle Position
        4: 'CO',   # Cut-Off
        5: 'BTN'   # Button
    }
    
    @staticmethod
    def to_game_state(obs: Dict, env: Any, player_id: Optional[int] = None) -> GameState:
        """
        Convertit une observation RLCard en GameState.
        
        Args:
            obs: Observation dict retournée par env.step() ou env.reset()
                Structure typique:
                {
                    'obs': np.array,  # Features encodées
                    'legal_actions': [0, 1, 2],  # Indices d'actions
                    'raw_obs': {  # État brut
                        'hand': ['AS', 'KD'],
                        'public_cards': ['JH', '9C', '4S'],
                        'chips': [9500, 10200, ...],
                        'my_chips': 9500,
                        ...
                    },
                    'action_record': [...]  # Historique
                }
            
            env: Environnement RLCard (pour accéder à la config)
        
        Returns:
            GameState standardisé
        """

        

        raw_obs = obs.get('raw_obs', {})

        
        # === 1. Extraction des cartes ===
        hole_cards = RLCardAdapter._convert_cards(raw_obs.get('hand', []))
        board = RLCardAdapter._convert_cards(raw_obs.get('public_cards', []))
        
        # === 2. Détermination de la street ===
        street = RLCardAdapter._determine_street(len(board))
        player_id = None
        # === 3. Position du joueur ===
        if player_id is None:
            if 'player_id' in raw_obs:
                player_id = raw_obs['player_id']
            elif 'current_player' in raw_obs:
                player_id = raw_obs['current_player']
            elif env is not None and hasattr(env, 'get_player_id'):
                player_id = env.get_player_id()
            else:
                player_id = 0  # Fallback
                print("⚠️  player_id introuvable, utilisation de 0 par défaut")


        num_players = env.num_players
        position = RLCardAdapter._get_position(player_id, num_players)
        
        # === 4. Extraction des blinds ===
        # RLCard stocke les blinds dans game.config
        game = env.game
        big_blind = getattr(game, 'big_blind', 100)
        small_blind = getattr(game, 'small_blind', 50)
        
        # === 5. Stacks et pot ===
        stack = RLCardAdapter.get_current_stack(raw_obs)
        
        # Pot = somme de toutes les mises
        all_chips = raw_obs.get('all_chips', [])
        if all_chips:
            # all_chips contient les chips de chaque joueur AVANT cette street
            initial_chips = sum(all_chips)
            current_chips = sum(raw_obs.get('chips', all_chips))
            pot_size = initial_chips - current_chips
        else:
            pot_size = raw_obs.get('pot', 0)
        
        # === 6. Joueurs actifs ===
        # Dans RLCard, on compte les joueurs avec chips > 0 et non fold
        chips = raw_obs.get('chips', [])
        num_active_players = sum(1 for c in chips if c > 0)
        
        # === 7. Amount to call ===
        amount_to_call = RLCardAdapter._calculate_amount_to_call(
            game,
            raw_obs,
            
        )
        
        # === 8. Actions légales ===
        legal_action_ids = obs.get('legal_actions', [])
        legal_actions = RLCardAdapter._convert_legal_actions(
            legal_action_ids,
            amount_to_call,
            stack
        )
        
        # === 9. Historique de la street ===
        actions_this_street = RLCardAdapter._extract_street_actions(
            obs.get('action_record', []),
            street
        )
        
        # === 10. Construction du GameState ===
        return GameState(
            hole_cards=hole_cards,
            board=board,
            street=street,
            position=position,
            num_active_players=num_active_players,
            pot_size=pot_size,
            stack=stack,
            big_blind=big_blind,
            small_blind=small_blind,
            amount_to_call=amount_to_call,
            legal_actions=legal_actions,
            actions_this_street=actions_this_street,
            player_id=player_id
        )
    
    # ═══════════════════════════════════════════════════════════
    # MÉTHODES PRIVÉES
    # ═══════════════════════════════════════════════════════════
    
    @staticmethod
    def _convert_cards(rlcard_cards: List[str]) -> List[str]:
        """
        Convertit les cartes RLCard en format standard.
        
        RLCard: ['AS', 'KD'] (uppercase, pas d'espace)
        Standard: ['As', 'Kd'] (capitalize)
        
        Args:
            rlcard_cards: Liste de cartes RLCard
        
        Returns:
            Liste de cartes au format standard
        """
        return [card[0].upper() + card[1].lower() for card in rlcard_cards]
    
    @staticmethod
    def _determine_street(num_board_cards: int) -> str:
        """
        Détermine la street depuis le nombre de cartes au board.
        
        Args:
            num_board_cards: 0, 3, 4 ou 5
        
        Returns:
            'preflop', 'flop', 'turn' ou 'river'
        """
        if num_board_cards == 0:
            return 'preflop'
        elif num_board_cards == 3:
            return 'flop'
        elif num_board_cards == 4:
            return 'turn'
        else:  # 5
            return 'river'
    
    @staticmethod
    def _get_position(player_id: int, num_players: int) -> str:
        """
        Convertit l'ID du joueur en position de table.
        
        RLCard numérote les joueurs de 0 à N-1 en partant de la SB.
        
        Args:
            player_id: ID du joueur (0-based)
            num_players: Nombre total de joueurs
        
        Returns:
            Position ('SB', 'BB', 'UTG', 'MP', 'CO', 'BTN')
        """
        if player_id is None:
            print("⚠️  player_id est None dans _get_position, utilisation de BTN par défaut")
            return 'BTN'  # Position neutre par défaut
        
        # Gestion du heads-up (2 joueurs)
        if num_players == 2:
            return 'SB' if player_id == 0 else 'BB'
        
        # 3-max à 6-max
        if num_players <= 6:
            if player_id < len(RLCardAdapter.POSITION_MAPPING):
                return RLCardAdapter.POSITION_MAPPING[player_id]
        
        # Tables > 6 joueurs : mapping simplifié
        positions = ['UTG', 'MP', 'MP', 'CO', 'BTN', 'SB', 'BB']
        offset = max(0, num_players - 6)
        adjusted_id = player_id - offset
        
        if 0 <= adjusted_id < len(positions):
            return positions[adjusted_id]
        
        return 'MP'
    
    @staticmethod
    def _calculate_amount_to_call(game, raw_obs: dict) -> float:
        """
        Calcule le montant à payer pour suivre (version robuste).
        """
        # try:
        #     # Méthode 1: Directement depuis raw_obs
        #     if not isinstance(raw_obs, dict):
        #         print('on est ici')
        #         logger.warning(f"raw_obs n'est pas un dict: {type(raw_obs)}")
        #         return 0.0
            
        #     if 'amount_to_call' in raw_obs:
        #         bb = raw_obs.get('stakes', [1, 2])[1]
        #         print(f'on est la, {bb}')
        #         if bb == 0:
        #             return 0.0
        #         return raw_obs['amount_to_call'] / bb
            
        #     # Méthode 2: Calculer depuis all_chips
        #     all_chips = raw_obs.get('all_chips', [])
        #     my_chips = raw_obs.get('my_chips', 0)
            
        #     if not all_chips:
        #         return 0.0
            
        #     # Stack initial standard
        #     initial_stack = 100
        #     max_invested = initial_stack - min(all_chips) if all_chips else 0
        #     my_invested = initial_stack - my_chips
            
        #     amount_to_call = max(0, max_invested - my_invested)
            
        #     # 🔧 FIX: Protection division par zéro
        #     bb = raw_obs.get('stakes', [1, 2])[1]
        #     if bb == 0:
        #         # logger.warning("BB = 0, retour 0.0 pour amount_to_call")
        #         return 0.0
            
        #     return amount_to_call / bb
            
        # except Exception as e:
        #     # logger.warning(f"Erreur calcul amount_to_call: {e}, retour 0")
        #     return 0.0
        try:
            # 1. Récupérer les mises actuelles de tout le monde sur ce tour
            # (Dans ton exemple : [0, 0, 0, 0, 1, 2])
            current_round_chips = raw_obs['all_chips']
            
            # 2. Trouver qui je suis
            my_seat = raw_obs['current_player']
            
            # 3. Ce que j'ai déjà mis (ex: 0)
            my_investment = current_round_chips[my_seat]
            
            # 4. La mise la plus haute à égaler (ex: 2)
            max_investment = max(current_round_chips)
            
            # 5. La différence
            diff = max_investment - my_investment
            
            return diff

        except KeyError:
            # Sécurité si les clés n'existent pas
            print('key error')
            return 0
        
    @staticmethod
    def get_current_stack(raw_obs) -> float:
        try:
            my_seat = raw_obs['current_player']
            all_stacks = raw_obs['stakes']
            my_stack = all_stacks[my_seat]
            return float(my_stack)
        except (KeyError, IndexError):
            return 0.0
    
    @staticmethod
    def _convert_legal_actions(
        legal_action_ids: List[int],
        amount_to_call: int,
        stack: int
    ) -> List[str]:
        """
        Convertit les IDs d'actions RLCard en noms d'actions.
        
        Args:
            legal_action_ids: Liste d'indices [0, 1, 2, ...]
            amount_to_call: Montant à call (pour différencier check/call)
            stack: Stack du joueur (pour all-in)
        
        Returns:
            Liste de noms ['fold', 'call', 'raise']
        """
        actions = []
        
        for action_id in legal_action_ids:
            action_name = RLCardAdapter.ACTION_MAPPING.get(action_id)
            
            if action_name == 'call' and amount_to_call == 0:
                # Call avec amount=0 → check
                action_name = 'check'
            
            if action_name == 'raise' and stack <= amount_to_call:
                # Raise impossible si on peut juste call/all-in
                continue
            
            if action_name:
                actions.append(action_name)
        
        # Fold est toujours possible (sauf si on peut check)
        if 'check' not in actions and 'fold' not in actions:
            actions.append('fold')
        
        return actions
    
    @staticmethod
    def _extract_street_actions(
        action_record: List,
        current_street: str
    ) -> List[str]:
        """
        Extrait les actions de la street courante depuis l'historique.
        
        RLCard stocke l'historique sous forme:
        [
            (player_id, 'call', amount),
            (player_id, 'raise', amount),
            ...
        ]
        
        Args:
            action_record: Historique complet RLCard
            current_street: Street actuelle
        
        Returns:
            Liste d'actions formatées ['call_100', 'raise_300', ...]
        """
        # RLCard ne sépare pas explicitement par street dans action_record
        # On doit déduire les limites depuis les cartes publiques
        
        # Simplifié pour l'instant: on retourne les N dernières actions
        # TODO: Améliorer avec détection des changements de street
        
        actions = []
        for record in action_record[-10:]:  # 10 dernières actions max
            if len(record) >= 3:
                player_id, action_name, amount = record[0], record[1], record[2]
                
                if amount > 0:
                    actions.append(f"{action_name}_{amount}")
                else:
                    actions.append(action_name)
        
        return actions
    
    @staticmethod
    def from_game_state(game_state: GameState, env: Any) -> int:
        """
        Convertit une action GameState (string) en action RLCard (int).
        
        Sens inverse: pour exécuter l'action décidée par l'agent.
        
        Args:
            game_state: État avec l'action choisie
            env: Environnement RLCard
        
        Returns:
            Action ID pour env.step(action_id)
        
        TODO: À implémenter quand on aura l'agent
        """
        raise NotImplementedError("Conversion GameState → RLCard action")


# ═══════════════════════════════════════════════════════════
# TESTS UNITAIRES
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import rlcard
    from rlcard.envs.nolimitholdem import NolimitholdemEnv
    
    print("=" * 70)
    print("🧪 TESTS RLCardAdapter")
    print("=" * 70)
    
    # Test 1: Preflop
    print("\n📊 Test 1: Preflop UTG")
    env = rlcard.make('no-limit-holdem', config={
        'game_num_players': 6,
        'small_blind': 50,
        'big_blind': 100
    })
    
    obs, player_id = env.reset()
    game_state = RLCardAdapter.to_game_state(obs, env)
    
    print(f"  {game_state}")
    print(f"  Hole cards: {game_state.hole_cards}")
    print(f"  Street: {game_state.street}")
    print(f"  Position: {game_state.position}")
    print(f"  Stack (BB): {game_state.effective_stack_bb:.1f}")
    print(f"  Amount to call (BB): {game_state.amount_to_call_bb:.1f}")
    print(f"  Legal actions: {game_state.legal_actions}")
    assert game_state.street == 'preflop'
    assert game_state.big_blind == 100
    print("  ✅ PASS")
    
    # Test 2: Après quelques actions
    print("\n📊 Test 2: Après call/raise")
    env.step(0)  # Player 0 call
    obs, player_id = env.step(1)  # Player 1 raise
    
    if not env.is_over():
        game_state = RLCardAdapter.to_game_state(obs, env)
        print(f"  {game_state}")
        print(f"  Actions this street: {game_state.actions_this_street}")
        print(f"  Amount to call: {game_state.amount_to_call}")
        print(f"  Pot odds: {game_state.pot_odds:.2%}")
        print("  ✅ PASS")
    
    # Test 3: Flop
    print("\n📊 Test 3: Flop (si atteint)")
    env2 = rlcard.make('no-limit-holdem')
    obs, _ = env2.reset()
    
    # Simuler jusqu'au flop (tous call)
    for _ in range(12):  # Force quelques tours
        if not env2.is_over():
            legal_actions = obs['legal_actions']
            action = legal_actions[0] if legal_actions else 0
            obs, _ = env2.step(action)
    
    if not env2.is_over():
        game_state = RLCardAdapter.to_game_state(obs, env2)
        if game_state.street != 'preflop':
            print(f"  {game_state}")
            print(f"  Board: {game_state.board}")
            print(f"  Street: {game_state.street}")
            print(f"  Pot (BB): {game_state.pot_size_bb:.1f}")
            print("  ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ TESTS TERMINÉS (adapter créé, validation avec RLCard réel)")
    print("=" * 70)
