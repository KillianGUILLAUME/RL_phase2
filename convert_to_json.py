import pickle
import xgboost as xgb
import os

# Chemins
pkl_path = 'models/xgb/xgb_pluribus_V1.pkl'
json_path = 'models/xgb/xgb_pluribus_V1.json'

print(f"🛠 Version XGBoost utilisée : {xgb.__version__}")

try:
    # 1. Chargement via Pickle
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Modèle chargé avec succès via Pickle.")

    # 2. Extraction du Booster (si c'est un wrapper Scikit-Learn)
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        print("ℹ️  Extraction du Booster depuis le wrapper Sklearn.")
    else:
        booster = model
        print("ℹ️  Booster natif détecté.")

    # 3. Sauvegarde en JSON
    booster.save_model(json_path)
    print(f"🎉 SUCCÈS ! Modèle converti : {json_path}")
    print("👉 Tu peux maintenant réinstaller XGBoost récent.")

except Exception as e:
    print(f"❌ Erreur : {e}")