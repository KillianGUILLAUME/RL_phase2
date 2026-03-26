import os
import glob
import torch
import numpy as np
from sb3_contrib import MaskablePPO
from features.feature_builder import FeatureExtractor_v2
from adapters.rlcard_adapter import RLCardAdapter

class PSROAgent:
    """
    A true Game Theory Optimal (GTO) approximation agent via Policy Space Response Oracles.
    This meta-agent selects one of its historical models drawn strictly from the mathematical 
    Nash Equilibrium probabilities calculated over the Empirical Payoff Matrix.
    """
    def __init__(self, env, archive_dir: str = "models/psro_archive/"):
        self.env = env
        self.archive_dir = archive_dir
        self.extractor = FeatureExtractor_v2()
        self.model_input_dim = len(self.extractor.get_feature_names())
        
        self.models = []
        self.nash_probs = []
        self._load_archive()
        
    def _load_archive(self):
        zip_files = sorted(glob.glob(os.path.join(self.archive_dir, "*.zip")))
        weights_path = os.path.join(self.archive_dir, "nash_weights.npy")
        
        if not zip_files:
            print(f"⚠️ Aucun modèle .zip trouvé dans {self.archive_dir}.")
            return
            
        if os.path.exists(weights_path):
            self.nash_probs = np.load(weights_path)
        else:
            print("⚠️ Fichier nash_weights.npy introuvable, fallback aux probabilités uniformes (1/N).")
            self.nash_probs = np.ones(len(zip_files)) / len(zip_files)
            
        print(f"⚖️ PSROAgent charge {len(zip_files)} modèles avec la distribution de Nash.")
        for i, file in enumerate(zip_files):
            # Only load models that have strictly > 0% probability to save RAM
            prob = self.nash_probs[i]
            if prob > 0.0001:
                try:
                    model = MaskablePPO.load(file, device='cpu')
                    self.models.append({
                        "name": os.path.basename(file),
                        "policy": model.policy,
                        "prob": prob
                    })
                    print(f"   ┣━ Loaded: {os.path.basename(file)} (Weight: {prob*100:.1f}%)")
                except Exception as e:
                    print(f"   ┣━ ❌ Failed to load {os.path.basename(file)}: {e}")
            else:
                print(f"   ┣━ Ignored: {os.path.basename(file)} (Weight: 0.0% - Inexploitable purge)")
                
        # Re-normalize just the loaded models to ensure sum=1.0 accurately
        loaded_probs = np.array([m["prob"] for m in self.models])
        loaded_probs /= loaded_probs.sum()
        for i, m in enumerate(self.models):
            m["prob"] = loaded_probs[i]
            
        print(f"   ┗━ Total active memory bank: {len(self.models)} policies.")

    def step(self, state):
        return self.eval_step(state)[0]

    def eval_step(self, state):
        if not self.models:
            raise ValueError("PSROAgent cannot act because its memory bank is empty.")
            
        # THE MAGIC OF PSRO: Sampling based on Nash Equilibrium probabilities
        probs = [m["prob"] for m in self.models]
        chosen_idx = np.random.choice(len(self.models), p=probs)
        policy = self.models[chosen_idx]["policy"]
        
        # 1. Adapt and extract
        game_state = RLCardAdapter.to_game_state(state, self.env)
        obs = self.extractor.extract(game_state).astype(np.float32)

        # 2. Action mask
        legal_actions = list(state['legal_actions'].keys())
        action_mask = np.zeros(self.env.num_actions, dtype=bool)
        action_mask[legal_actions] = True

        # 3. Predict
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).unsqueeze(0).to('cpu')
            action_mask_tensor = torch.tensor(action_mask).unsqueeze(0).to('cpu')
            
            action_tensor, _ = policy.predict(obs_tensor, action_masks=action_mask_tensor, deterministic=True)
            action = int(action_tensor.item())

        if action not in legal_actions:
            action = np.random.choice(legal_actions)

        return action, {}

    def __getstate__(self):
        return self.__dict__