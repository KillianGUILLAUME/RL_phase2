import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SOTAPokerNet(BaseFeaturesExtractor):
    """
    Le cerveau moderne : Pas de règles manuelles, que des Embeddings.
    """
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 1. Card Embeddings (La magie remplace ton FeatureExtractor)
        # 52 cartes + 1 (carte cachée/vide) -> vecteur de taille 64
        self.card_embedding = nn.Embedding(53, 64)
        
        # 2. Traitement du Board (LSTM ou Transformer est le vrai SOTA)
        # Pour faire simple ici : MLP
        self.board_processor = nn.Sequential(
            nn.Linear(5 * 64, 128), # 5 cartes du board * 64 dims
            nn.ReLU()
        )
        
        # 3. Traitement de la Main
        self.hand_processor = nn.Sequential(
            nn.Linear(2 * 64, 128), # 2 cartes en main
            nn.ReLU()
        )
        
        # 4. Infos contextuelles (Pot, Stack) - Pas besoin d'embedding
        self.context_processor = nn.Sequential(
            nn.Linear(10, 32), # Pot, Stack, Dealer pos, etc.
            nn.ReLU()
        )
        
        # Fusion
        self.final_layer = nn.Linear(128 + 128 + 32, features_dim)

    def forward(self, observations):
        # On extrait les indices des cartes (envoyés par RLCard)
        hand_idx = observations['hand']   # ex: [12, 4] (As, 5)
        board_idx = observations['board'] # ex: [51, 0, 1, 52, 52] (K, 2, 3, vide, vide)
        context = observations['context'] # ex: [100, 50...] (Pot, Stack)

        # La couche d'Embedding fait le travail de compréhension des cartes
        hand_emb = self.card_embedding(hand_idx).flatten(1)
        board_emb = self.card_embedding(board_idx).flatten(1)
        
        # Passage dans les sous-réseaux
        h_out = self.hand_processor(hand_emb)
        b_out = self.board_processor(board_emb)
        c_out = self.context_processor(context)
        
        # Fusion
        return self.final_layer(torch.cat([h_out, b_out, c_out], dim=1))

# Lancement de l'entrainement
model = PPO("MultiInputPolicy", env, policy_kwargs={"features_extractor_class": SOTAPokerNet}, verbose=1)
model.learn(total_timesteps=10_000_000)