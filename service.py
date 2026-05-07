import bentoml
import torch

from src.model import BloodCellCNN


# Nazwy klas w zbiorze BloodMNIST
# Model zwraca liczbę od 0 do 7, a ta lista zamienia ją na nazwę klasy
CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "immature granulocytes",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platelet",
]


@bentoml.service
class BloodCellClassifier:
    def __init__(self):
        # Użycie GPU, jeśli CUDA jest dostępna, w przeciwnym razie CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Wczytanie modelu wytrenowanego w zadaniu 1 z checkpointa
        self.model = BloodCellCNN.load_from_checkpoint(
            "checkpoints/best.ckpt",
            map_location=self.device,
        )

        # Przeniesienie modelu na wybrane urządzenie i ustawienie trybu predykcji
        self.model.to(self.device)
        self.model.eval()

    @bentoml.api
    def predict(self, image: list) -> dict:
        # Zamiana listy otrzymanej przez API na tensor PyTorch
        x = torch.tensor(image, dtype=torch.float32)

        # Dodanie wymiaru batcha
        # Klient wysyła jeden obraz o kształcie [3, 28, 28]
        # Model oczekuje batcha o kształcie [1, 3, 28, 28]
        x = x.unsqueeze(0)

        # Przeniesienie danych wejściowych na to samo urządzenie co model
        x = x.to(self.device)

        # Wyłączenie liczenia gradientów, ponieważ wykonujemy predykcję, a nie trening
        with torch.no_grad():
            logits = self.model(x)

            # Zamiana surowych wyników modelu na prawdopodobieństwa klas
            probabilities = torch.softmax(logits, dim=1)

            # Wybranie klasy z największym prawdopodobieństwem
            predicted_class_id = int(torch.argmax(probabilities, dim=1).item())

        # Zwrócenie odpowiedzi w formacie JSON
        return {
            "device": str(self.device),
            "predicted_class_id": predicted_class_id,
            "predicted_class_name": CLASS_NAMES[predicted_class_id],
            "probabilities": probabilities[0].cpu().tolist(),
        }