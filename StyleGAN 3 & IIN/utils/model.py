import pickle


def load_model(model_path, device="cuda"):
    with open(model_path, "rb") as f:
        models = pickle.load(f)
    return models["G_ema"].to(device), models["D"].to(device)

    