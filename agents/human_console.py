class HumanConsoleAgent:
    def __init__(self, num_actions):
        self.use_raw = True
        self.num_actions = num_actions

    def step(self, state):
        # --- Affichage (inchangé) ---
        print("\n" + "="*50)
        print(f"👤 C'EST À TOI DE JOUER ! (Joueur {state['raw_obs']['current_player']})")
        print("-" * 50)
        
        raw = state['raw_obs']
        print(f"🃏 Ta Main : {raw['hand']}")
        # On affiche proprement les stacks
        my_stack = raw['stakes'][raw['current_player']]
        print(f"💰 Ton Stack: {my_stack} | Pot: {raw['pot']}")
        print(f"📜 Tapis : {raw['public_cards']}")
        
        # --- Gestion des Actions ---
        print("\nActions possibles :")
        # On récupère les objets Action réels (Enums)
        raw_legal_actions = state['raw_legal_actions']
        
        action_map = {
            0: "FOLD",
            1: "CHECK/CALL",
            2: "RAISE HALF POT",
            3: "RAISE POT",
            4: "ALL-IN"
        }
        
        # On affiche les options
        valid_indices = []
        for action_enum in raw_legal_actions:
            idx = action_enum.value
            valid_indices.append(idx)
            print(f"  [{idx}] {action_map.get(idx, 'Unknown')}")
            
        # Boucle de saisie
        while True:
            try:
                choice = input("\n👉 Choisis ton action (numéro) : ")
                action_int = int(choice)
                
                if action_int in valid_indices:
                    # === LA CORRECTION EST ICI ===
                    # On ne retourne pas l'int, mais l'objet Action correspondant
                    # On cherche dans raw_legal_actions celui qui a la bonne valeur
                    chosen_action_enum = next(a for a in raw_legal_actions if a.value == action_int)
                    return chosen_action_enum
                else:
                    print(f"❌ Action {action_int} non valide. Choisis parmi {valid_indices}")
            except ValueError:
                print("❌ Entre un chiffre.")

    def eval_step(self, state):
        return self.step(state), {}