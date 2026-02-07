from matplotlib.path import Path
import os
import pickle


def save_model(model, model_name, directory="./models"):
    # Créer le dossier s'il n'existe pas
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Chemin du fichier
    filepath = os.path.join(directory, f"{model_name}.pkl")

    # Sauvegarder
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Modèle sauvegardé: {filepath}")
    print(f"   Taille: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB\n")

    return filepath


def load_model(model_name, directory="./models"):
    filepath = os.path.join(directory, f"{model_name}.pkl")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modèle introuvable: {filepath}")

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Modèle chargé: {filepath}\n")

    return model
