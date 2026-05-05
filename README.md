# BloodMNIST Classification with PyTorch Lightning

## Opis projektu

Projekt dotyczy klasyfikacji typów komórek krwi na podstawie obrazów mikroskopowych ze zbioru **BloodMNIST**.

Celem było przygotowanie prostego pipeline'u MLOps dla modelu:

- przygotowanie danych przez `Dataset` / `DataLoader`,
- użycie `LightningDataModule`,
- użycie `LightningModule`,
- monitoring treningu w WandB,
- sprawdzenie, czy model faktycznie się uczy przez porównanie `train_loss` i `val_loss`,
- optymalizacja hiperparametrów za pomocą Optuny.

## Dataset

Dataset: **BloodMNIST**  
Źródło: https://medmnist.com/
BloodMNIST zawiera obrazy mikroskopowe komórek krwi.

Zadanie: ``` klasyfikacja wieloklasowa obrazów ```

Liczba klas:``` 8```

Klasy:

```text
0 → basophil
1 → eosinophil
2 → erythroblast
3 → immature granulocytes
4 → lymphocyte
5 → monocyte
6 → neutrophil
7 → platelet
```

Kształt danych wejściowych: ```[batch_size, 3, 28, 28]```

Obrazy są kolorowe, mają 3 kanały RGB i rozmiar `28 x 28` pikseli.
BloodMNIST ma gotowy podział na zbiór treningowy, walidacyjny i testowy, dane nie były dzielone ręcznie.

## Struktura projektu

```text
zadanie1/
├── src/
│   ├── datamodule.py
│   ├── model.py
│   ├── train.py
│   └── tune.py
├── requirements.txt
├── README.md
├── .gitignore
├── optuna_results_50epochs.csv
└── optuna_results_lr_weight_decay_50epochs.csv
```

## Pliki projektu

### `src/datamodule.py`
Odpowiada za przygotowanie danych:
- pobranie danych BloodMNIST,
- przygotowanie transformacji obrazów,
- utworzenie zbiorów `train`, `validation` i `test`,
- utworzenie DataLoaderów.
batch_size = 128

### `src/model.py`
Plik zawiera model w PyTorch Lightning. Model ```BloodCellCNN``` to prosta sieć CNN do klasyfikacji obrazów. Architektura:
```text
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Flatten → Linear → ReLU → Dropout → Linear
```
Model przyjmuje dane o kształcie``` [batch_size, 3, 28, 28]```.

Model zwraca wynik o kształcie ``` [batch_size, 8]```.

Funkcja straty to ```cross_entropy```.

Optymalizator: ```AdamW```

Logowane metryki:

```text
train_loss
train_acc
val_loss
val_acc
test_loss
test_acc
```
Metryki treningowe i walidacyjne były logowane raz na epokę.

### `src/train.py`
Uruchamia trening modelu. Zawiera:

- utworzenie DataModule,
- utworzenie modelu,
- logger WandB,
- checkpoint najlepszego modelu,
- EarlyStopping,
- testowanie najlepszego checkpointa.

Trening zatrzymuje się, jeśli przez 5 kolejnych epok `val_loss` się nie poprawia (EarlyStopping: ``` patience = 5```).

### `src/tune.py`

Plik odpowiada za optymalizację hiperparametrów poprzez Optuna. Optymalizowane hiperparametry: ``` learning_rate, 
weight_decay```.

Optuna minimalizowała ``` best val_loss```, czyli najlepszą stratę walidacyjną uzyskaną podczas treningu.


## Instalacja środowiska

Projekt był przygotowany na Windowsie z użyciem środowiska `.venv`, które trzeba stworzyć lokalnie i zainstalować zależności z `requirements.txt`.

### Windows / PowerShell

Utworzenie środowiska:

```powershell
py -3.11 -m venv .venv
```

Aktywacja środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalacja zależności:

```powershell
pip install -r requirements.txt
```

### Linux / macOS

Utworzenie środowiska:

```bash
python3 -m venv .venv
```

Aktywacja środowiska:

```bash
source .venv/bin/activate
```

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Logowanie do WandB:

```powershell
wandb login
```

## Uruchamianie

Test DataModule:

```powershell
python src\datamodule.py
```

Test modelu:

```powershell
python src\model.py
```

Trening:

```powershell
python src\train.py
```

Optymalizacja hiperparametrów:

```powershell
python src\tune.py
```

## Eksperymenty


## Eksperymenty

Początkowo model był trenowany przez 10 epok. Ten etap służył głównie do sprawdzenia, czy cały pipeline działa poprawnie: dane się ładują, model się uczy, a WandB zapisuje metryki.

Pierwszy baseline dla 10 epok osiągnął `best val_loss = 0.2722` oraz `test_acc = 0.877`. Następnie użyto Optuny do optymalizacji samego `learning_rate`. Optuna znalazła `learning_rate ≈ 0.003857`, co poprawiło wynik krótkiego treningu do `best val_loss = 0.2567` i `test_acc = 0.886`.

Po analizie wykresów trening został wydłużony do maksymalnie 50 epok. Dodano `EarlyStopping` z `patience = 5`, aby trening zatrzymywał się automatycznie, gdy `val_loss` przestaje się poprawiać.

Dla 50 epok baseline z `learning_rate = 0.001` i `weight_decay = 0.0001` osiągnął `best val_loss = 0.1874` oraz `test_acc = 0.9336`. Znaleziony wcześniej `learning_rate` dla krótszego treningu nie działał lepiej przy dłuższym treningu, model z `learning_rate ≈ 0.003857` osiągnął `best val_loss = 0.2329` oraz `test_acc = 0.8906`.

Następnie powtórzono optymalizację dla konfiguracji z 50 epokami. Optymalizacja samego `learning_rate` nie poprawiła baseline, najlepsze wyniki wyniosły `learning_rate ≈ 0.002335` i `best val_loss = 0.1808`, a baseline miał `best val_loss = 0.1726`.

Na końcu zoptymalizowano dwa hiperparametry jednocześnie: `learning_rate` i `weight_decay`. Przeszukiwane zakresy to `learning_rate` od `0.0001` do `0.01` oraz `weight_decay` od `0.000001` do `0.01`. Wykonano 10 prób.

Najlepszy zestaw hiperparametrów znaleziony przez Optunę:

- `learning_rate = 0.0015435312235389442`
- `weight_decay = 0.00001053809851988176`

Wynik Optuny dla tych hiperparametrów to `best val_loss = 0.1653`, czyli poprawa względem baseline o `0.0184`.

Finalny trening z tymi hiperparametrami osiągnął `best val_loss = 0.1728` oraz `test_acc = 0.9196`.


## Porównanie wyników

| Eksperyment | learning_rate | weight_decay | best epoch | stopped epoch | best val_loss | test_acc |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 10 epok | 0.001 | 0.0001 | 9 | 9 | 0.2722 | 0.877 |
| Optuna LR 10 epok | 0.003857 | 0.0001 | 8 | 9 | 0.2567 | 0.886 |
| Baseline 50 epok | 0.001 | 0.0001 | 23 | 28 | 0.1874 | 0.9336 |
| Optuna LR 50 epok | 0.003857 | 0.0001 | 15 | 20 | 0.2329 | 0.8906 |
| Optuna LR + WD 50 epok | 0.0015435 | 0.0000105 | 16 | 21 | 0.1728 | 0.9196 |


Optuna poprawiła wynik walidacyjny, szczególnie po jednoczesnej optymalizacji `learning_rate` i `weight_decay`. Najlepszy wynik walidacyjny po finalnym treningu z hiperparametrami z Optuny wyniósł `best val_loss = 0.1728`. Najwyższy wynik testowy uzyskał baseline trenowany maksymalnie przez 50 epok: `test_acc = 0.9336`. Optymalizacja poprawiła `val_loss`, ale nie poprawiła końcowej accuracy na zbiorze testowym.




## Końcowy wniosek

Finalny model po Optunie osiągnął:

```text
best val_loss = 0.1728
test_acc = 0.9196
```

Dla porównania baseline 50 epok osiągnął:

```text
best val_loss = 0.1874
test_acc = 0.9336
```

Optuna poprawiła więc wynik walidacyjny, ale najwyższą skuteczność na zbiorze testowym uzyskał baseline 50 epok.
