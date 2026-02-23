import torch
import numpy as np
from sb3_contrib import MaskablePPO
import onnxruntime as ort
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_builder import FeatureExtractor_v2
from adapters.rlcard_adapter import RLCardAdapter

class OnnxablePolicy(torch.nn.Module):
    """ Un petit wrapper pour isoler la partie prédiction du réseau """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        features = self.policy.extract_features(observation)
        latent_pi, _ = self.policy.mlp_extractor(features)
        logits = self.policy.action_net(latent_pi)
        return logits

def export_model_to_onnx(model_path="champion_10M_steps.zip", output_path="poker_bot_v1.onnx", input_dim=203):
    from typing import Callable
    def constant_schedule(value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return value
        return func

    custom_objects = {
        "learning_rate": constant_schedule(5e-5),
        "lr_schedule": constant_schedule(5e-5),
        "clip_range": constant_schedule(0.15)
    }
    
    try:
        model = MaskablePPO.load(model_path, device="cpu", custom_objects=custom_objects)
    except Exception:
        model = MaskablePPO.load(model_path, device="cpu")
        
    onnx_policy = OnnxablePolicy(model.policy)

    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        onnx_policy,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=['observation'],
        output_names=['logits']
    )
    print(f"✅ Modèle ONNX exporté avec succès vers {output_path} !")
class ONNXPokerBot:
    def __init__(self, onnx_model_path: str, env):
        self.env = env
        self.extractor = FeatureExtractor_v2()
        self.onnx_model_path = onnx_model_path
        self.session = None  # Lazy loading crucial for SubprocVecEnv multiprocessing
        self.input_name = None

    def __getstate__(self):
        """ Évite de pickler la session C++ (inéluctable erreur avec SubprocVecEnv) """
        state = self.__dict__.copy()
        state['session'] = None
        return state

    def _initialize_session(self):
        """ 🚀 Lancement du moteur ONNX (en mode CPU) """
        self.session = ort.InferenceSession(
            self.onnx_model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def step(self, state):
        if self.session is None:
            self._initialize_session()
            
        # 1. Extraction (vos 203 variables)
        game_state = RLCardAdapter.to_game_state(state, self.env)
        obs = self.extractor.extract(game_state)
        
        # Format attendu par ONNX : float32 avec dimension Batch (1, 203)
        obs_array = np.expand_dims(obs.astype(np.float32), axis=0)

        # ⚡ 2. INFÉRENCE ONNX (Vitesse maximale) ⚡
        # Le réseau nous renvoie les scores bruts des 4 actions
        logits = self.session.run(None, {self.input_name: obs_array})[0][0]

        # 3. Gestion des masques en NumPy pur (Le secret !)
        masks = np.zeros(self.env.num_actions, dtype=bool)
        legal_actions = list(state['legal_actions'].keys())
        for a in legal_actions:
            masks[a.value if hasattr(a, 'value') else int(a)] = True

        # On met un score infiniment négatif (-inf) aux actions illégales
        # Ainsi, l'agent ne pourra jamais les choisir
        logits[~masks] = -np.inf
        
        # 4. On prend l'action légale avec le plus gros score
        action_idx = int(np.argmax(logits))

        # 5. Mapping et retour (comme d'habitude)
        raw_legal = state.get('raw_legal_actions', [])
        for a in raw_legal:
            if a.value == action_idx:
                return a
                
        return raw_legal[0] if raw_legal else action_idx
        
    def eval_step(self, state):
        return self.step(state), {}

if __name__ == "__main__":
    import sys
    # Usage CLI pour exporter : python agents/onnx_policy.py my_model.zip out.onnx
    if len(sys.argv) >= 3:
        export_model_to_onnx(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python agents/onnx_policy.py <input.zip> <output.onnx>")