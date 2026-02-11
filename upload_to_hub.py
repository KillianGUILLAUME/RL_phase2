"""
Upload de modèles RL vers Hugging Face Hub.

Fonctions principales :
  - convert_model_to_zip(model_path) : Convertit un modèle (.pkl, .pt, .json, etc.) en .zip
  - upload_to_hub(zip_path, repo_name)  : Upload le .zip vers Hugging Face Hub

Usage :
    python models/upload_to_hub.py --model models/rl/best_model/checkpoint.pt --repo "mon-user/poker-agent-v1"

Prérequis :
    pip install huggingface_hub
    huggingface-cli login  # Authentification (token HF)
"""

import os
import sys
import zipfile
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
except ImportError:
    print("❌ huggingface_hub non installé. Lance : pip install huggingface_hub")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 1. CONVERSION MODÈLE → ZIP
# ═══════════════════════════════════════════════════════════════

def convert_model_to_zip(model_path: str, output_dir: str = None) -> str:
    """
    Convertit un fichier modèle (.pkl, .pt, .json, .ubj, dossier de checkpoint)
    en une archive .zip prête pour l'upload.

    Args:
        model_path: Chemin vers le modèle (fichier ou dossier)
        output_dir: Dossier de sortie pour le .zip (défaut : même dossier que le modèle)

    Returns:
        Chemin absolu vers le fichier .zip créé

    Exemples:
        # Fichier unique
        zip_path = convert_model_to_zip("models/xgb/xgb_pluribus_v1.pkl")
        # → models/xgb/xgb_pluribus_v1.zip

        # Dossier de checkpoint
        zip_path = convert_model_to_zip("models/rl/best_model/")
        # → models/rl/best_model.zip
    """
    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Modèle introuvable : {model_path}")

    # Déterminer le nom et le dossier de sortie
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du zip (sans extension d'origine)
    if model_path.is_file():
        zip_name = model_path.stem + ".zip"
    else:
        zip_name = model_path.name + ".zip"

    zip_path = output_dir / zip_name

    print(f"📦 Conversion en .zip...")
    print(f"   Source  : {model_path}")
    print(f"   Sortie  : {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if model_path.is_file():
            # Fichier unique → on l'ajoute à la racine du zip
            zf.write(model_path, model_path.name)
            print(f"   ✅ Fichier ajouté : {model_path.name}")
        elif model_path.is_dir():
            # Dossier → on ajoute tout le contenu récursivement
            file_count = 0
            for file in model_path.rglob("*"):
                if file.is_file() and not file.name.startswith('.'):
                    arcname = file.relative_to(model_path)
                    zf.write(file, arcname)
                    file_count += 1
            print(f"   ✅ {file_count} fichiers ajoutés")
        else:
            raise ValueError(f"❌ Type de chemin non supporté : {model_path}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"   📏 Taille : {size_mb:.2f} MB")

    return str(zip_path)


# ═══════════════════════════════════════════════════════════════
# 2. UPLOAD VERS HUGGING FACE HUB
# ═══════════════════════════════════════════════════════════════

def upload_to_hub(
    zip_path: str,
    repo_name: str,
    commit_message: str = None,
    private: bool = False,
    model_card: dict = None,
    token: str = None
) -> str:
    """
    Upload un fichier .zip de modèle vers Hugging Face Hub.

    Args:
        zip_path: Chemin vers le fichier .zip à uploader
        repo_name: Nom du repo HF (ex: "mon-user/poker-agent-v1")
        commit_message: Message de commit (auto-généré si None)
        private: Si True, le repo est privé
        model_card: Dict avec les infos pour le README du repo (optionnel)
        token: Token HF (utilise le token CLI si None)

    Returns:
        URL du repository sur Hugging Face

    Exemples:
        url = upload_to_hub(
            "models/xgb/xgb_pluribus_v1.zip",
            "killianguillaume/poker-xgboost-v1"
        )
    """
    zip_path = Path(zip_path).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"❌ Fichier .zip introuvable : {zip_path}")

    if not zip_path.suffix == '.zip':
        raise ValueError(f"❌ Le fichier doit être un .zip, reçu : {zip_path.suffix}")

    api = HfApi(token=token)

    # === 1. Créer le repo (ou le récupérer s'il existe déjà) ===
    print(f"\n🚀 Upload vers Hugging Face Hub")
    print(f"   Repo   : {repo_name}")
    print(f"   Fichier: {zip_path.name}")

    try:
        repo_url = create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token
        )
        print(f"   ✅ Repo créé/trouvé : {repo_url}")
    except Exception as e:
        print(f"   ⚠️ Erreur création repo : {e}")
        raise

    # === 2. Créer un Model Card (README.md) pour le repo ===
    readme_content = _generate_model_card(repo_name, zip_path, model_card)

    # Créer un dossier temporaire avec le zip + README
    tmp_dir = Path("/tmp") / f"hf_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copier le zip
        shutil.copy2(zip_path, tmp_dir / zip_path.name)

        # Écrire le Model Card
        readme_path = tmp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        # Écrire les métadonnées
        metadata = {
            "upload_date": datetime.now().isoformat(),
            "source_file": str(zip_path),
            "zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
            "framework": "RL_phase2",
            "task": "poker-nlhe-6max"
        }
        with open(tmp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # === 3. Upload tout le dossier ===
        if commit_message is None:
            commit_message = f"Upload model {zip_path.name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message,
            token=token
        )

        hub_url = f"https://huggingface.co/{repo_name}"
        print(f"\n   🎉 Upload terminé !")
        print(f"   🔗 {hub_url}")

        return hub_url

    finally:
        # Nettoyage du dossier temporaire
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _generate_model_card(repo_name: str, zip_path: Path, extra_info: dict = None) -> str:
    """Génère un README.md (Model Card) pour le repo HF."""

    model_name = repo_name.split("/")[-1] if "/" in repo_name else repo_name
    size_mb = zip_path.stat().st_size / (1024 * 1024)

    card = f"""---
tags:
  - reinforcement-learning
  - poker
  - no-limit-holdem
  - xgboost
  - dqn
  - ppo
license: cc-by-nc-4.0
library_name: rl_phase2
pipeline_tag: reinforcement-learning
---

# 🃏 {model_name}

Agent de poker IA pour le **No-Limit Texas Hold'em 6-max**, entraîné dans le cadre du projet [RL_phase2](https://github.com/killianguillaume/RL_phase2).

> ⚠️ **AVERTISSEMENT** : Ce modèle est **STRICTEMENT réservé à la recherche académique** en intelligence artificielle et théorie des jeux.
> Toute utilisation commerciale ou à des fins de triche est **formellement interdite**.

## 🚫 Usages interdits

| Usage | Statut |
|---|---|
| Triche ou aide en temps réel sur des sites de poker en ligne | **❌ INTERDIT** |
| Exploitation commerciale (vente, SaaS, API payante) | **❌ INTERDIT** |
| Contournement des CGU des plateformes de poker | **❌ INTERDIT** |
| Développement de bots jouant avec de l'argent réel | **❌ INTERDIT** |
| Recherche académique, éducation, expérimentation personnelle | ✅ Autorisé |
| Publication scientifique (avec citation) | ✅ Autorisé |

Les plateformes de poker en ligne (PokerStars, GGPoker, Winamax, etc.) **interdisent explicitement** les outils d'aide à la décision en temps réel.
Tout contrevenant s'expose à la **fermeture de compte** et à des **poursuites légales**.

---

## Détails du modèle

| Propriété | Valeur |
|---|---|
| **Fichier** | `{zip_path.name}` |
| **Taille** | {size_mb:.2f} MB |
| **Date** | {datetime.now().strftime('%Y-%m-%d')} |
| **Framework** | RL_phase2 (PyTorch / XGBoost / SB3) |
| **Jeu** | No-Limit Hold'em 6-max |
| **Features** | 87 dimensions (cartes, position, stack, GTO) |
| **Licence** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

## Usage

```python
# Télécharger et charger le modèle
from huggingface_hub import hf_hub_download
import zipfile

# Téléchargement
zip_path = hf_hub_download(
    repo_id="{repo_name}",
    filename="{zip_path.name}"
)

# Extraction
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("./models/downloaded/")
```
"""

    if extra_info:
        card += "\n## Informations supplémentaires\n\n"
        for key, value in extra_info.items():
            card += f"- **{key}** : {value}\n"

    card += """
## 📄 Licence

Ce modèle est distribué sous la licence **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.fr)** (Attribution - Pas d'Utilisation Commerciale).

| Vous pouvez | Vous ne pouvez PAS |
|---|---|
| ✅ Utiliser pour la recherche | ❌ Vendre ou commercialiser |
| ✅ Modifier et adapter | ❌ Créer un service payant |
| ✅ Partager (avec attribution) | ❌ Utiliser pour tricher au poker |
| ✅ Publier des travaux dérivés (non-commerciaux) | ❌ Retirer la mention d'attribution |

## 🙏 Citation & Remerciements

Ce travail repose notamment sur les données de **Pluribus** (Meta AI Research) :

> Brown, N., & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.

Données accessibles via le projet [Poker-Hand-History](https://github.com/uoftcprg/poker-hand-history) (University of Toronto CPRG).

---

*Recherche en mathématiques appliquées / IA — Killian GUILLAUME*
"""

    return card


# ═══════════════════════════════════════════════════════════════
# 3. FONCTION PRINCIPALE (TOUT-EN-UN)
# ═══════════════════════════════════════════════════════════════

def convert_and_upload(
    model_path: str,
    repo_name: str,
    private: bool = False,
    token: str = None,
    model_card: dict = None,
    commit_message: str = None
) -> str:
    """
    Fonction tout-en-un : convertit un modèle en .zip et l'upload sur HF Hub.

    Args:
        model_path: Chemin vers le modèle (fichier .pkl/.pt/.json ou dossier)
        repo_name: Nom du repo HF (ex: "killianguillaume/poker-agent-v1")
        private: Repo privé ou public
        token: Token HF (optionnel, utilise le token CLI sinon)
        model_card: Infos supplémentaires pour le Model Card
        commit_message: Message de commit custom

    Returns:
        URL du repo Hugging Face

    Usage:
        url = convert_and_upload(
            model_path="models/xgb/xgb_pluribus_v1.pkl",
            repo_name="killianguillaume/poker-xgboost-v1",
            private=True
        )
    """
    print("=" * 60)
    print("🤗 UPLOAD MODÈLE → HUGGING FACE HUB")
    print("=" * 60)

    # Étape 1 : Conversion
    zip_path = convert_model_to_zip(model_path)

    # Étape 2 : Upload
    url = upload_to_hub(
        zip_path=zip_path,
        repo_name=repo_name,
        private=private,
        token=token,
        model_card=model_card,
        commit_message=commit_message
    )

    print("\n" + "=" * 60)
    print(f"✅ TERMINÉ — Modèle disponible sur : {url}")
    print("=" * 60)

    return url


# ═══════════════════════════════════════════════════════════════
# 4. CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🤗 Upload un modèle RL vers Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Upload un modèle XGBoost
  python models/upload_to_hub.py --model models/xgb/xgb_v1.pkl --repo "mon-user/poker-xgb-v1"

  # Upload un checkpoint DQN (dossier)
  python models/upload_to_hub.py --model models/rl/best_model/ --repo "mon-user/poker-dqn-v2"

  # Upload un modèle PPO (SB3 .zip déjà existant)
  python models/upload_to_hub.py --model models/ppo_final.zip --repo "mon-user/poker-ppo-v1" --private

  # Avec un token spécifique
  python models/upload_to_hub.py --model models/rl/best.pt --repo "mon-user/poker-v3" --token "hf_xxxx"
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Chemin vers le modèle (fichier .pkl/.pt/.json/.zip ou dossier)"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        required=True,
        help='Nom du repo HF (ex: "mon-user/poker-agent-v1")'
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Créer un repo privé"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Token Hugging Face (utilise le token CLI si omis)"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Message de commit custom"
    )

    args = parser.parse_args()

    # Si le fichier est déjà un .zip, on skip la conversion
    model_path = Path(args.model)
    if model_path.suffix == '.zip' and model_path.is_file():
        print("📦 Fichier .zip détecté, skip conversion.")
        upload_to_hub(
            zip_path=str(model_path),
            repo_name=args.repo,
            private=args.private,
            token=args.token,
            commit_message=args.message
        )
    else:
        convert_and_upload(
            model_path=args.model,
            repo_name=args.repo,
            private=args.private,
            token=args.token,
            commit_message=args.message
        )
