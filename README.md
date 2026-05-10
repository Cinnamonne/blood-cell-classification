## 1. BloodMNIST Classification with PyTorch Lightning


## Opis projektu

Projekt dotyczy klasyfikacji typów komórek krwi na podstawie obrazów mikroskopowych ze zbioru **BloodMNIST**.

Celem było przygotowanie procesu trenowania modelu:
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
Lightning module, zawiera model w PyTorch Lightning. Model ```BloodCellCNN``` to prosta sieć CNN do klasyfikacji obrazów. Architektura:
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

Logowane metryki: ```train_loss, train_acc, val_loss, val_acc, test_loss, test_acc```

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

Projekt był przygotowany na Windowsie z użyciem środowiska `.venv`.

PyTorch i torchvision są instalowane osobno, projekt używa wersji z obsługą CUDA 12.8. Pozostałe biblioteki są instalowane z pliku `requirements.txt`.

### Windows / PowerShell

Utworzenie środowiska:

```powershell
py -3.11 -m venv .venv
```

Aktywacja środowiska:

```powershell
.\.venv\Scripts\Activate.ps1
```

Aktualizacja `pip`:

```powershell
python -m pip install --upgrade pip
```

Instalacja PyTorch z obsługą CUDA 12.8:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Instalacja pozostałych zależności:

```powershell
python -m pip install -r requirements.txt
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

Aktualizacja `pip`:

```bash
python -m pip install --upgrade pip
```

Instalacja PyTorch z obsługą CUDA 12.8:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Instalacja pozostałych zależności:

```bash
python -m pip install -r requirements.txt
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

Początkowo model był trenowany przez 10 epok. Pierwszy baseline dla 10 epok osiągnął `best val_loss = 0.2722` oraz `test_acc = 0.877`. Użyto Optuny do optymalizacji `learning_rate`. Optuna znalazła `learning_rate ≈ 0.003857`, co poprawiło wynik do `best val_loss = 0.2567` i `test_acc = 0.886`.

Po analizie wykresów trening został wydłużony do maksymalnie 50 epok. Dodano `EarlyStopping` z `patience = 5`, aby trening zatrzymywał się automatycznie, gdy `val_loss` przestaje się poprawiać.

Dla 50 epok baseline z `learning_rate = 0.001` i `weight_decay = 0.0001` osiągnął `best val_loss = 0.1874` oraz `test_acc = 0.9336`, więc wyniki się poprawiły.

Znaleziony wcześniej `learning_rate` dla krótszego treningu nie działał lepiej przy 50 epokach, model z `learning_rate ≈ 0.003857` osiągnął `best val_loss = 0.2329` oraz `test_acc = 0.8906`.

Następnie powtórzono optymalizację samego `learning_rate` dla konfiguracji z 50 epokami. Najlepsza próba Optuny dała `learning_rate ≈ 0.002335` i `best val_loss = 0.1808`, ale baseline z `learning_rate = 0.001` miał lepszy wynik walidacyjny, dlatego sama optymalizacja `learning_rate` nie poprawiła baseline. Jedną z przyczyn mogła być mała liczba prób.

Na końcu zoptymalizowano dwa hiperparametry jednocześnie: `learning_rate` i `weight_decay`. Przeszukiwane zakresy to `learning_rate` od `0.0001` do `0.01` oraz `weight_decay` od `0.000001` do `0.01`. Wykonano 10 prób.

Najlepszy zestaw hiperparametrów znaleziony przez Optunę to `learning_rate = 0.0015435312235389442` i `weight_decay = 0.00001053809851988176`. Wynik Optuny dla tych hiperparametrów to `best val_loss = 0.1653`, czyli poprawa względem baseline o `0.0184`. Ostateczny trening z tymi hiperparametrami osiągnął `best val_loss = 0.1728` oraz `test_acc = 0.9196`.


## Porównanie wyników

| Eksperyment                   | learning_rate | weight_decay | best epoch | stopped epoch | best val_loss | test_acc |
|-------------------------------|---:|---:|---:|---:|---:|---:|
| Baseline 10 epok              | 0.001 | 0.0001 | 9 | 9 | 0.2722 | 0.877 |
| Optuna LR 10 epok             | 0.003857 | 0.0001 | 8 | 9 | 0.2567 | 0.886 |
| Baseline 50 epok              | 0.001 | 0.0001 | 23 | 28 | 0.1874 | 0.9336 |
| LR z Optuny 10 epok + 50 epok | 0.003857 | 0.0001 | 15 | 20 | 0.2329 | 0.8906 |
| Optuna LR + WD 50 epok        | 0.0015435 | 0.0000105 | 16 | 21 | 0.1728 | 0.9196 |

Optuna poprawiła `best val_loss`, szczególnie po jednoczesnej optymalizacji `learning_rate` i `weight_decay`. Najlepszy wynik walidacyjny uzyskano dla `Optuna LR+WD 50 epok`: `best val_loss = 0.1728`. Najwyższy wynik testowy uzyskał `Baseline 50 epok`: `test_acc = 0.9336`. Optymalizacja dla 50 epok poprawiła `val_loss`, ale nie poprawiła `test_acc` na zbiorze testowym.

## Końcowy wniosek

Ostateczny model Optuna LR + WD 50 epok osiągnął:

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



------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

## 2.  BentoML model serving

Ta część projektu uruchamia wytrenowany klasyfikator BloodMNIST jako usługę REST API poprzez BentoML. Używany jest modelu wytrenowanego i wczytuje checkpoint z ```checkpoints/best.ckpt```.

Kod API znajduje się w: ```service.py```, przykładowy klient wysyłający zapytania do API znajduje się w ```client.py```

Kod pomocniczy do wyświetlania obrazów z BloodMNIST znajduje się w pliku: ```show_bloodmnist.py```

### Wymagany checkpoint

Przed uruchomieniem serwera trzeba umieścić wybrany checkpoint modelu w:```checkpoints/best.ckpt```

Struktura projektu:

```text
blood-cell-classification/
├── checkpoints/
│   └── best.ckpt
├── service.py
├── client.py
├── show_bloodmnist.py
├── requirements.txt
└── README.md
```

### Uruchomienie serwisu BentoML

W pierwszym terminalu należy uruchomić serwer:

```powershell
bentoml serve service:BloodCellClassifier
```

Serwer działa lokalnie pod adresem: ```http://localhost:3000```

Adres do wysyłania obrazów do predykcji:```http://localhost:3000/predict```

Klient wysyła tam obraz poprzez `POST`, a serwer zwraca przewidzianą klasę komórki krwi.
Serwis używa GPU, jeśli CUDA jest dostępna, inaczej działa na CPU.

### Uruchomienie klienta

W drugim terminalu należy aktywować środowisko wirtualne i uruchomić jeden z poniższych wariantów.

uruchominie dla pierwszego przykładu ze zbioru testowego BloodMNIST:

```powershell
python client.py
```

Można wybrać dowolną liczbę losowych przykładów, przykładowo chcąc 4 losowe przykłady ze zbioru testowego:

```powershell
python client.py 4
```

Wybranie określonych przykładów ze zbioru testowego BloodMNIST według indeksów:

```powershell
python client.py --indices 0 10 25
```

Wybranie pierwszego przykładu BloodMNIST i wyświetlenie obrazu po predykcji:

```powershell
python client.py --show-images
```

Użycie 4 losowych przykładów BloodMNIST i wyświetlenie ich po predykcji:

```powershell
python client.py 4 --show-images
```

Użycie wybranych przykładów BloodMNIST i wyświetlenie ich po predykcji:

```powershell
python client.py --indices 0 10 25 --show-images
```

Użycie jednego własnego obrazu z pliku:

```powershell
python client.py sample_images\monocyte.jpg
```

Użycie wszystkich obsługiwanych obrazów z folderu:

```powershell
python client.py sample_images
```

Obsługiwane formaty: ``` .png, .jpg, .jpeg, .bmp, .tiff, .webp ```

Opcja `--show-images` działa dla przykładów z BloodMNIST. Ustawiony jest limit liczby wyświetlanych obrazów, żeby nie tworzyć zbyt dużego wykresu.

### Przykładowy wynik

Przykładowa komenda:

```powershell
python client.py
```

Przykładowa odpowiedź:

```text
Source: BloodMNIST test_dataset[0]
True label: 3
Response:
{
  "device": "cuda",
  "predicted_class_id": 5,
  "predicted_class_name": "monocyte",
  "probabilities": [
    0.0030919541604816914,
    0.0000002918598625001323,
    0.000017613718227948993,
    0.1857094168663025,
    0.00003775582445086911,
    0.8111426830291748,
    0.00000032874950761652144,
    0.000000000001300972242261611
  ]
}
```

Zwracane jest urządzenie użyte do wykonania predykcji, przewidziany identyfikator klasy, nazwę przewidzianej klasy oraz prawdopodobieństwa dla wszystkich 8 klas BloodMNIST.


------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
## 3. Cloud VM deployment

Ta część projektu uruchamia serwer BentoML w kontenerze Docker na maszynie wirtualnej w chmurze Azure.

```text
Model: BloodCellCNN wytrenowany na BloodMNIST
Serving framework: BentoML
Konteneryzacja: Docker
Cloud VM: Azure Virtual Machine z Ubuntu
Test request: test_deployed_service.py
```

Kod serwera: ``` service.py ```

Konfiguracja kontenera: ``` Dockerfile ```

Skrypt testujący serwer: ``` test_deployed_service.py ```

### Budowanie obrazu Docker

Obraz Docker można zbudować lokalnie albo bezpośrednio na maszynie wirtualnej:

```powershell
docker build -t blood-cell-classifier .
```

Na maszynie wirtualnej, jeśli Docker wymaga uprawnień administratora:

```bash
sudo docker build -t blood-cell-classifier .
```

### Uruchomienie kontenera

Lokalnie:

```powershell
docker run --rm -p 3000:3000 blood-cell-classifier
```

Na maszynie wirtualnej:

```bash
sudo docker run --rm -p 3000:3000 blood-cell-classifier
```

Kontener uruchamia serwer BentoML na porcie: ``` 3000 ```

Adres do wysyłania zapytań do predykcji ma postać:

```text
http://PUBLIC_IP:3000/predict
```

### Uruchomienie na Azure VM

Serwis został uruchomiony na maszynie wirtualnej Azure z systemem Ubuntu.

Użyty publiczny endpoint:

```text
http://68.221.141.213:3000/predict
```

Na maszynie wirtualnej wykonano:

```bash
git clone https://github.com/Cinnamonne/blood-cell-classification.git
cd blood-cell-classification
sudo docker build -t blood-cell-classifier .
sudo docker run --rm -p 3000:3000 blood-cell-classifier
```

W ustawieniach sieciowych Azure dodano regułę inbound dla portu```3000```.

### Test serwera

Do testowania publicznego endpointu służy: ``` test_deployed_service.py ```

Domyślnie skrypt wysyła zapytanie do serwera działającego na Azure VM:
```text
http://68.221.141.213:3000/predict
```

Przykładowe uruchomienie:

```powershell
python test_deployed_service.py
```

Można też podać endpoint ręcznie, jeśli publiczny adres VM się zmieni:

```powershell
python test_deployed_service.py http://NEW_PUBLIC_IP:3000/predict
```

Przykładowy wynik:

```text
Endpoint: http://68.221.141.213:3000/predict
Response:
{
  'device': 'cpu',
  'predicted_class_id': 7,
  'predicted_class_name': 'platelet',
  'probabilities': [
    0.008859595283865929,
    0.08915380388498306,
    0.11022865027189255,
    0.08168857544660568,
    0.014438780955970287,
    0.013760769739747047,
    0.021206164732575417,
    0.6606636047363281
  ]
}
```

### Informacja o uruchomieniu przed sprawdzeniem

Przed sprawdzeniem zadania trzeba ponownie uruchomić VM w Azure Portal, połączyć się z nią przez SSH i uruchomić kontener:

```bash
cd blood-cell-classification
sudo docker run --rm -p 3000:3000 blood-cell-classifier
```

Po uruchomieniu kontenera endpoint można ponownie sprawdzić komendą:

```powershell
python test_deployed_service.py
```


------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------

## 4. Serverless deployment

Ta część projektu uruchamia predykcję modelu w architekturze serverless na Azure. Funkcja działa automatycznie po dodaniu obrazu do Azure Blob Storage.

Wykorzystane elementy:

```text
Model: BloodCellCNN wytrenowany na BloodMNIST
Cloud storage: Azure Blob Storage
Serverless function: Azure Function uruchomiona jako kontener
Container registry: Azure Container Registry
Docker image: blood-cell-function:v1
```

### Struktura rozwiązania

W Azure Storage Account utworzone zostały dwa kontenery: ``` input-images``` i  ```results```. Kontener `input-images` służy do wrzucania obrazów wejściowych, kontener `results` służy do zapisywania wyników predykcji w formacie JSON.

Przebieg działania wygląda tak:

```text
upload obrazu do input-images
→ automatyczne uruchomienie Azure Function
→ pobranie obrazu przez funkcję
→ wykonanie predykcji modelem BloodCellCNN
→ zapis wyniku jako JSON do results
```

### Kod funkcji

Kod funkcji serverless znajduje się w folderze ```azure_function/```. Najważniejsze pliki:

```text
azure_function/function_app.py
azure_function/Dockerfile
azure_function/requirements.txt
azure_function/host.json
```

Plik `function_app.py` zawiera funkcję `PredictBloodCell`, która reaguje na dodanie pliku do kontenera `input-images`.

Funkcja wczytuje obraz, przygotowuje go do formatu oczekiwanego przez model, wykonuje predykcję i zapisuje wynik do kontenera `results`.

### Docker image

Funkcja została spakowana do obrazu Docker. Obraz zawiera:

```text
kod funkcji
kod modelu
checkpoint modelu
biblioteki potrzebne do predykcji
```
Budowanie obrazu lokalnie:

```powershell
docker buildx build --platform linux/amd64 --provenance=false -f azure_function/Dockerfile -t bloodcellh4acr.azurecr.io/blood-cell-function:v1 --push .
```

Obraz został wysłany do Azure Container Registry: ```bloodcellh4acr.azurecr.io```

### Uruchomienie na Azure

W Azure utworzone zostały:

```text
Storage Account
Azure Container Registry
Azure Function uruchomiona w Container Apps
```

Azure Function została uruchomiona z obrazu Docker: ```bloodcellh4acr.azurecr.io/blood-cell-function:v1```

Funkcja używa triggera Blob Storage. Oznacza to, że uruchamia się automatycznie po dodaniu nowego obrazu do kontenera `input-images`.

### Test działania

Do kontenera `input-images` został wrzucony obraz testowy. W kontenerze `results` pojawił się plik JSON z wynikiem predykcji:

```json
{
  "source_blob": "input-images/basophil.png",
  "predicted_class_id": 7,
  "predicted_class_name": "platelet",
  "probabilities": [
    0.008859595283865929,
    0.08915380388498306,
    0.11022865027189255,
    0.08168857544660568,
    0.014438780955970287,
    0.013760769739747047,
    0.021206164732575417,
    0.6606636047363281
  ]
}
```