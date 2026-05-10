import torch
from torch import nn
import torch.nn.functional as F

import lightning as L
from torchmetrics.classification import MulticlassAccuracy


class BloodCellCNN(L.LightningModule):
    """
    Model CNN do klasyfikacji obrazów komórek krwi z BloodMNIST.
    Model dostaje obraz 28x28 RGB i zwraca 8 wyników, po jednym dla każdej klasy komórki krwi.
    """

    def __init__(
        self,
        num_classes: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()


        self.save_hyperparameters() #zapisuje hiperparametry modelu, by Lightning i WandB mogły je pokazać w logach

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


        #część konwolucyjna modelu - wyciąga cechy z obrazu
        self.features = nn.Sequential(
            #obraz wejściowy ma 3 kanały, warstwa tworzy 32 mapy cech
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            #zmniejsza obraz z 28x28 do 14x14.
            nn.MaxPool2d(kernel_size=2),

            #druga warstwa konwolucyjna
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            #zmniejsza obraz z 14x14 do 7x7
            nn.MaxPool2d(kernel_size=2),

            #trzecia warstwa konwolucyjna
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            #zmniejsza obraz z 7x7 do 3x3
            nn.MaxPool2d(kernel_size=2),
        )


        #część klasyfikacyjna modelu - zamienia cechy obrazu na wynik klasyfikacji
        self.classifier = nn.Sequential(
            #zamienia tensor na jeden wektor
            nn.Flatten(),

            #po części konwolucyjnej mamy 128 kanałów i rozmiar 3x3
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),

            #dropout, by ograniczyć przeuczanie
            nn.Dropout(0.3),

            #ostatnia warstwa zwraca 8 wyników klas
            nn.Linear(128, num_classes),
        )

        #metryki dla treningu, walidacji i testu
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        """
        Przepływ danych przez model.

        Wejście:
        [batch_size, 3, 28, 28]

        Wyjście:
        [batch_size, 8]
        """

        x = self.features(x)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        """
        Jeden krok treningowy.
        Lightning wywołuje tę metodę dla każdego batcha treningowego.
        """

        images, labels = batch

        logits = self(images) #model oblicza wyniki dla każdej klasy (wew. jest forward)

        loss = F.cross_entropy(logits, labels) #cross entropy - standardowa strata dla klasyfikacji wieloklasowej

        preds = torch.argmax(logits, dim=1) #wybieramy klasę z najwyższym wynikiem

        acc = self.train_acc(preds, labels) #liczymy accuracy treningowe

        # Logowanie po epoce
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Jeden krok walidacyjny.
        Walidacja sprawdza model na danych niewykorzystywanych do aktualizacji wag
        """

        images, labels = batch

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Jeden krok testowy
        Test wykonujemy po treningu na osobnym zbiorze testowym
        """

        images, labels = batch

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Definiuje optymalizator
        AdamW aktualizuje wagi modelu i używa weight_decay jako regularyzacji
        """

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer


if __name__ == "__main__":
    # Szybki test czy model przyjmuje obrazy o dobrym kształcie

    model = BloodCellCNN(num_classes=8)

    fake_images = torch.randn(128, 3, 28, 28)
    outputs = model(fake_images)

    print("Shape wyjścia modelu:", outputs.shape)