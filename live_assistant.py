import rlcard
from rlcard.games.nolimitholdem.round import Action
from agents.xgboost_agent import XGBoostRLCardAgent as XGBoostAgent
import sys
import pprint

# ==========================================
# 🛠️ VERSION FINALE - ROBUSTE
# ==========================================

class LiveAssistant:
    def __init__(self):
        print("\n🤖 INITIALISATION DU CERVEAU XGBOOST...")
        self.dummy_env = rlcard.make('no-limit-holdem', config={'game_num_players': 6})
        self.agent = XGBoostAgent('models/xgb/xgb_pluribus_V1.pkl', env=self.dummy_env)
        self.configure_table()
        self.reset_hand()

    def configure_table(self):
        print("\n" + "="*40)
        print("⚙️  CONFIGURATION DE LA TABLE")
        print("="*40)
        try:
            np_input = input(f"Nombre de joueurs ? (Entrée = 6) : ")
            self.num_players = int(np_input) if np_input else 6

            bb_input = input(f"Big Blind ? (Entrée = 2) : ")
            self.big_blind = int(bb_input) if bb_input else 2

            stack_input = input(f"Ton Stack de départ ? (Entrée = 100) : ")
            self.my_stack = int(stack_input) if stack_input else 100
            
            print(f"✅ Table prête : {self.num_players} Joueurs | BB={self.big_blind}")
        except ValueError:
            print("❌ Entre des nombres entiers !")
            self.configure_table()

    def reset_hand(self):
        self.my_hand = []
        self.board = []
        self.pot = 0
        self.stage = 0 
        # Position par défaut : Bouton (Dernier)
        self.my_position = self.num_players - 1 

    def set_position(self):
        print("\nOù es-tu assis ?")
        print(f"0 = SB (Small Blind)")
        print(f"1 = BB (Big Blind)")
        print(f"{self.num_players - 1} = BTN (Bouton/Dealer - Meilleure place)")
        try:
            pos = input(f"Ta position (0-{self.num_players-1}) : ")
            self.my_position = int(pos)
            print(f"📍 Position enregistrée : {self.my_position}")
        except:
            print("❌ Position invalide. On reste sur l'ancienne.")

    def parse_cards(self, input_str):
        """
        Notation internationale
        as : as spade -> as de pique
        kh : king heart -> roi de coeur
        qd : queen diamond -> dame de carreau
        jc : jack club : valet de trèfle

        """
        # Nettoyage robuste : vire les virgules, points, etc.
        clean = input_str.replace(',', ' ').replace(';', ' ').replace('-', ' ').upper().split()
        # Remplacement 10 -> T pour éviter les erreurs
        final = [c.replace('10', 'T') for c in clean]
        return final

    def get_legal_actions_enum(self):
        print("\nActions autorisées ?")
        print(" [1] Standard (Fold, Check, Bet/Raise)")
        print(" [2] Face à une mise (Fold, Call, Raise)")
        print(" [3] Crise (Fold, Call, All-in)")
        print(" [4] Check/All-in")
        
        choice = input("👉 Choix (1-4) : ")
        actions_str = []
        
        if choice == '1': actions_str = ['fold', 'check', 'raise', 'allin']
        elif choice == '2': actions_str = ['fold', 'call', 'raise', 'allin']
        elif choice == '3': actions_str = ['fold', 'call', 'allin']
        elif choice == '4': actions_str = ['check', 'allin']
        else: actions_str = ['fold', 'check', 'raise'] # Fallback
        
        mapping = {
            'fold': Action.FOLD, 'check': Action.CHECK_CALL, 'call': Action.CHECK_CALL,
            'raise': Action.RAISE_POT, 'bet': Action.RAISE_POT, 'allin': Action.ALL_IN
        }
        return list(set([mapping[a] for a in actions_str if a in mapping]))

    def ask_bot(self):
        try:
            # Saisie rapide du Pot
            p_in = input(f"💰 Pot Total ? (Entrée={self.pot}) : ")
            if p_in: self.pot = int(p_in)
            
            # Saisie rapide du Stack (optionnel)
            s_in = input(f"Ton Stack ? (Entrée={self.my_stack}) : ")
            if s_in: self.my_stack = int(s_in)
            
            raw_legal = self.get_legal_actions_enum()
            
            # Construction State
            all_chips = [self.my_stack] * self.num_players
            all_chips[self.my_position] = self.my_stack
            
            mock_state = {
                'player_id': self.my_position,
                'current_player': self.my_position,

                'legal_actions': {a.value: None for a in raw_legal},
                'raw_legal_actions': raw_legal,
                'raw_obs': {
                    'hand': self.my_hand, 
                    'public_cards': self.board,
                    'all_chips': all_chips, 
                    'my_chips': self.my_stack,
                    'stakes': [0]*self.num_players, 
                    'pot': self.pot,
                    'current_player': self.my_position, 
                    'stage': self.stage,
                    'legal_actions': raw_legal,
                    'player_id': self.my_position
                }
            }
            print("\n🧠 ANALYSE...")
            action, info = self.agent.eval_step(mock_state)
            print(action,info)
            # AFFICHAGE VISUEL
            print("\n" + "█"*40)
            a_str = str(action).replace('Action.', '')
            if "FOLD" in a_str: print(f"🛑  COUCHE-TOI (FOLD)")
            elif "CHECK" in a_str or "CALL" in a_str: print(f"✅  CHECK / PAYE (CALL)")
            elif "RAISE" in a_str: print(f"🚀  RELANCE (RAISE)")
            elif "ALL_IN" in a_str: print(f"💣  ALL-IN (TAPIS) !")
            print("█"*40)
            
            print("\n📊 Confiance :\n")
            labels = {0:"FOLD", 1:"CHECK/CALL", 2:"RAISE", 3:"RAISE POT", 4:"ALL-IN"}
            for k,v in sorted(info.items(), key=lambda x:x[1], reverse=True):
                if v > 0.02: print(f"   {labels.get(k,k):<10} : {v*100:.1f}%")
            print("\n")
            
        except Exception as e:
            print(f"❌ Oups, erreur : {e}")

    def run(self):
        print("\n🎰 ASSISTANT POKER LIVE (FINAL)")
        print("Commandes : 'Ah Kd' (Cartes), 'flop', 'turn', 'river', 'pos' (Position), 'new', 'q'")
        
        while True:
            try:
                state_txt = f"Main:{self.my_hand} | Board:{self.board} | Pos:{self.my_position}"
                cmd = input(f"\n({state_txt}) 👉 : ").lower().strip()
                
                if cmd == 'q': break
                elif cmd == 'new': 
                    self.reset_hand()
                    print("🔄 Nouvelle main !")
                elif cmd == 'config': self.configure_table()
                elif cmd == 'pos': self.set_position()
                
                elif cmd == 'flop':
                    c = input("Cartes du Flop ? : ")
                    self.board = self.parse_cards(c)
                    self.stage = 1
                elif cmd == 'turn':
                    c = input("Carte Turn ? : ")
                    self.board.extend(self.parse_cards(c))
                    self.stage = 2
                elif cmd == 'river':
                    c = input("Carte River ? : ")
                    self.board.extend(self.parse_cards(c))
                    self.stage = 3
                
                elif cmd == '' or cmd == 'go':
                    if not self.my_hand: print("⚠️ Rentre tes cartes d'abord !"); continue
                    self.ask_bot()
                    
                else:
                    # On suppose que c'est les cartes
                    parsed = self.parse_cards(cmd)
                    if parsed:
                        self.my_hand = parsed
                        print(f"👌 Main : {self.my_hand}")
            except KeyboardInterrupt:
                print("\nAu revoir !")
                break
            except Exception as e:
                print(f"⚠️ Erreur de saisie ({e}), réessaie.")

if __name__ == "__main__":
    LiveAssistant().run()