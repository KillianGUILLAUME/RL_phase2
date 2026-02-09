from datetime import datetime


import rlcard
from rlcard.games.nolimitholdem.round import Action
from agents.xgboost_agent import XGBoostRLCardAgent as XGBoostAgent
import sys
import re

# On utilise rich pour une belle interface (pip install rich)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    print("❌ Installe 'rich' pour une belle interface : pip install rich")
    sys.exit()

console = Console()

class LiveAssistantPro:
    def __init__(self):
        console.print("[bold green]🤖 CHARGEMENT DU CERVEAU XGBOOST...[/bold green]")
        self.dummy_env = rlcard.make('no-limit-holdem', config={'game_num_players': 6})
        self.agent = XGBoostAgent('models/xgb/xgb_pluribus_V1.json', env=self.dummy_env)
        
        # État du jeu
        self.num_players = 6
        self.my_position = 5  # Bouton par défaut
        self.reset_hand()

    def reset_hand(self):
        self.my_hand = []
        self.board = []
        self.pot = 0
        self.my_stack = 100
        self.stage = 0  # 0: Preflop, 1: Flop, 2: Turn, 3: River
        self.last_decision = None
        self.last_probs = {}
        self.amount_to_call = 0

    def format_card(self, card_str):
        """Transforme 'Ah' en 'A♥' coloré"""
        suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
        colors = {'S': 'white', 'H': 'red', 'D': 'cyan', 'C': 'green'}
        
        c = card_str.upper().replace('10', 'T')
        if len(c) < 2: return c
        
        rank, suit = c[:-1], c[-1]
        if suit in suits:
            return Text(f"{rank}{suits[suit]}", style=f"bold {colors[suit]}")
        return Text(card_str)

    def get_dashboard(self):
        """Crée le tableau de bord visuel"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="board", size=5),
            Layout(name="info", size=3),
            Layout(name="decision", ratio=1)
        )

        # Header
        layout["header"].update(Panel(Text("🎰 LIVE POKER ASSISTANT - XGBOOST", justify="center", style="bold magenta"), box=box.HEAVY))

        # Board & Hand
        hand_view = Text("MA MAIN: ", style="bold yellow")
        for c in self.my_hand: hand_view.append(self.format_card(c)); hand_view.append(" ")
        
        board_view = Text(" | BOARD: ", style="bold blue")
        if not self.board: board_view.append(" (Vide)")
        for c in self.board: board_view.append(self.format_card(c)); board_view.append(" ")

        layout["board"].update(Panel(Text.assemble(hand_view, board_view), title="Situation", border_style="blue"))

        # Info
        pos_names = {0:"SB", 1:"BB", 2:"UTG", 3:"MP", 4:"CO", 5:"BTN"}
        info_txt = f"💰 Pot: {self.pot} | 💸 A payer: {self.amount_to_call} | 👤 Stack: {self.my_stack} | 📍 {pos_names.get(self.my_position, self.my_position)}"
        layout["info"].update(Panel(info_txt, style="white on black"))

        # Decision Table
        if self.last_decision:
            table = Table(
                title=f"🤖 CONSEIL: {self.last_decision}", 
                caption=f"[dim]{self.last_update}[/dim]", # <--- ICI
                box=box.SIMPLE
            )
            table.add_column("Action", justify="left")
            table.add_column("Confiance", justify="right")
            table.add_column("Barre", justify="left", style="magenta")

            labels = {0:"FOLD", 1:"CHECK/CALL", 2:"RAISE", 3:"ALL-IN"}
            
            # Trier par probabilité
            sorted_probs = sorted(self.last_probs.items(), key=lambda x: x[1], reverse=True)
            
            for k, v in sorted_probs:
                if v < 0.01: continue # On cache les probas < 1%
                bar_len = int(v * 20)
                bar = "█" * bar_len
                style = "green" if k == 1 else "red" if k == 0 else "yellow"
                table.add_row(labels.get(k, str(k)), f"{v*100:.1f}%", f"[{style}]{bar}[/{style}]")
            
            layout["decision"].update(Panel(table, border_style="green"))
        else:
            layout["decision"].update(Panel(Text("En attente de données...\nTapez cartes (ex: AhKd) ou 'go' pour analyser", justify="center"), border_style="dim"))

        return layout

    def smart_parse(self, user_input):
        """Détecte intelligemment l'intention de l'utilisateur"""
        raw = user_input.strip()
        if not raw: return "go" # Entrée vide = Analyser

        parts = raw.split()
        cmd = parts[0].lower()

        # Commandes système
        if cmd in ['q', 'quit']: return 'quit'
        if cmd in ['n', 'new', 'reset']: return 'new'
        
        # Commandes Pot / Stack / Pos
        if cmd.startswith('p'): # p 500 -> Pot = 500
            try: 
                val = int(parts[1])
                self.pot = val
                return "go"
            except: pass

        if cmd.startswith('s'): 
            try: 
                val = int(parts[1])
                self.my_stack = val
                return "go"
            except: pass
            
        if cmd.startswith('pos'): # pos 5 -> BTN
            try:
                self.my_position = int(parts[1])
                return "go"
            except: pass

        if cmd.startswith('b') or cmd.startswith('c'): 
            try: 
                val = int(parts[1])
                self.amount_to_call = val
                return "go"
            except: pass

        # Détection de cartes (Regex simple pour 2 chars ou plus)
        # Ex: Ah Kd ou Th9s ou AhKd
        clean_cards = raw.replace(',', ' ').upper()
        # Séparer les cartes collées (ex: AhKd -> Ah Kd)
        clean_cards = re.sub(r'([2-9TJQKA][SHDC])', r' \1 ', clean_cards).split()
        
        valid_cards = []
        for c in clean_cards:
            if re.match(r'^[2-9TJQKA][SHDC]$', c):
                # Remplacer T par T (déjà fait) ou 10 par T
                valid_cards.append(c)
        
        if valid_cards:
            if len(valid_cards) == 2 and not self.my_hand:
                self.my_hand = valid_cards
                return "go"
            elif len(valid_cards) >= 1:
                # Si on a déjà une main, c'est le board
                self.board.extend(valid_cards)
                # Auto-detect stage
                if len(self.board) == 3: self.stage = 1
                elif len(self.board) == 4: self.stage = 2
                elif len(self.board) == 5: self.stage = 3
                return "go" # Lance l'analyse auto après ajout board
            
        return "go"

    def run_inference(self):
        """Lance l'IA"""
        if not self.my_hand:
            self.last_decision = "⚠️ CARTES MANQUANTES"
            return

        # Construction des actions légales par défaut
        # Mappings: 0:Fold, 1:Check/Call, 2:Raise, 3:Allin
        raw_legal_enum = [Action.FOLD, Action.CHECK_CALL, Action.RAISE_POT, Action.ALL_IN]
        legal_vals = {0: None, 1: None, 2: None, 3: None} 

        # Construction du state RLCard
        all_chips = [self.my_stack] * self.num_players
        
        # On s'assure que la position est bien un int
        pid = int(self.my_position)

        stakes = [0] * self.num_players
        
        dealer_pos = (pid - 1) % self.num_players

        action_record = []

        if self.amount_to_call > 0:
            # On met la mise sur le premier adversaire venu pour simuler l'agression
            villain_idx = (pid - 1) % self.num_players
            stakes[villain_idx] = float(self.amount_to_call)
            
            # On ajoute l'action dans l'historique (C'est ÇA qui change tout)
            # ('raise' est le mot clé que RLCard comprend)
            action_record.append((villain_idx, 'raise'))

        

        mock_state = {
            'player_id': pid,           # 👈 IMPORTANT : ID au niveau racine
            'raw_legal_actions': raw_legal_enum,
            'legal_actions': legal_vals,
            'raw_obs': {
                'hand': self.my_hand, 
                'public_cards': self.board,
                'all_chips': all_chips, 
                'my_chips': self.my_stack,
                'stakes': stakes, 
                'pot': self.pot,
                'stage': self.stage,
                'legal_actions': raw_legal_enum,
                'player_id': pid,       # 👈 IMPORTANT : ID dans raw_obs
                'current_player': pid,   # 👈 IMPORTANT : ID du joueur actif
                'dealer_pos': dealer_pos, 
                'action_record': action_record, # <--- ICI LE CERVEAU SE DÉBLOQUE
                'big_blind': 2
            }
        }
        import pprint
        pprint.pprint(mock_state)
        try:
            print('ici')
            # ⏱️ ON MET A JOUR L'HEURE JUSTE AVANT LE CALCUL
            now = datetime.now().strftime("%H:%M:%S.%f")[:-3] # Heure:Min:Sec.ms
            
            # Appel au cerveau
            action, probs = self.agent.eval_step(mock_state)
            
            # Mise à jour des textes
            a_str = str(action).replace('Action.', '')
            if "FOLD" in a_str: res = "🛑 FOLD"
            elif "CHECK" in a_str or "CALL" in a_str: res = "✅ CHECK / CALL"
            elif "RAISE" in a_str: res = "🚀 RAISE"
            elif "ALL_IN" in a_str: res = "💣 ALL-IN"
            else: res = a_str
            
            self.last_decision = res
            self.last_probs = probs
            
            # 👇 PREUVE DE CALCUL
            # On affiche l'heure + les valeurs critiques utilisées
            self.last_update = f"Calculé à {now} | Inputs: Pot={self.pot}, Stack={self.my_stack}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.last_decision = f"ERREUR: {str(e)}"

    def loop(self):
        while True:
            # 1. Afficher l'interface
            console.clear()
            console.print(self.get_dashboard())
            
            # 2. Saisie utilisateur
            user_in = console.input("\n[bold cyan]Commande > [/bold cyan]")
            
            # 3. Traitement
            action = self.smart_parse(user_in)
            
            if action == 'quit': break
            if action == 'new': self.reset_hand()
            if action == 'go': self.run_inference()
            # 'update' ne fait rien de spécial, la boucle recommence et réaffiche

if __name__ == "__main__":
    app = LiveAssistantPro()
    app.loop()