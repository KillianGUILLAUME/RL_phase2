import pickle
import xgboost as xgb
import sys

print(f"🐍 Python: {sys.version}")
print(f"📦 XGBoost version: {xgb.__version__}")

path = 'models/xgb/xgb_pluribus_V1.pkl'
print(f"📂 Tentative de chargement de : {path}")

try:
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Chargement RÉUSSI !")
except EOFError:
    print("❌ Fichier corrompu ou vide.")
except Exception as e:
    print(f"❌ Erreur Python : {e}")