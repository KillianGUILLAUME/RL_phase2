import numpy as np

class PokerRewardShaper:
    """
    Transforme les gains bruts du poker en un signal normalisé pour PPO.
    Évite l'explosion des gradients et le 'Mode Collapse'.
    """
    def __init__(self, scale_factor=100.0, max_clip=3.0):
        # scale_factor = 100 signifie qu'une cave entière vaut 1.0
        self.scale_factor = scale_factor
        # max_clip = 3.0 signifie qu'on capte les gains extrêmes à +3.0 / -3.0
        self.max_clip = max_clip

    def shape(self, raw_payoff: float) -> float:
        """
        Applique le scaling et le clipping sur la récompense de fin de main.
        """
        if raw_payoff == 0:
            return 0.0
            
        # 1. Scaling : -100 jetons devient -1.0 | +221 jetons devient +2.21
        scaled_reward = raw_payoff / self.scale_factor
        
        # 2. Clipping : Sécurité au cas où un pot de 6 joueurs atteint +500
        clipped_reward = np.clip(scaled_reward, -self.max_clip, self.max_clip)
        
        return float(clipped_reward)