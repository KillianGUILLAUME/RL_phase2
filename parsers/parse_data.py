import zipfile
import io

phh_zip = "data/poker-hand-histories.zip"
fichier_a_inspecter = None

print(f"--- Recherche d'un fichier .phh dans {phh_zip} ---")

try:
    with zipfile.ZipFile(phh_zip, 'r') as zf:

        # Cherchons le premier fichier .phh pour l'inspecter
        for nom_fichier in zf.namelist():
            if nom_fichier.endswith('.phh'):
                fichier_a_inspecter = nom_fichier
                print(f"Fichier trouvé pour inspection : {fichier_a_inspecter}")
                break # On a trouvé notre "cobaye", on arrête de chercher

        if fichier_a_inspecter:
            print(f"\n--- Inspection des 20 premières lignes de {fichier_a_inspecter} ---")

            with zf.open(fichier_a_inspecter, 'r') as f_binaire:
                f_texte = io.TextIOWrapper(f_binaire, encoding='utf-8')

                for i in range(20):
                    try:
                        ligne = f_texte.readline()
                        if not ligne:
                            break
                        if ligne.strip():
                            print(f"Ligne {i+1}: {ligne.strip()}")
                        else:
                            print(f"Ligne {i+1}: [Ligne vide]")
                    except Exception as e:
                        print(f"Erreur de lecture (encodage?) à la ligne {i+1}: {e}")
                        break

            print("\n--- Fin de l'inspection ---")
        else:
            print("ERREUR : Aucun fichier .phh n'a été trouvé dans le .zip !")

except FileNotFoundError:
    print(f"Erreur : Le fichier '{phh_zip}' n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur générale est survenue : {e}")