import io
import json
import os
import sys
from pathlib import Path

import azure.functions as func
import torch
from azure.storage.blob import BlobServiceClient
from PIL import Image
from torchvision import transforms


#dodanie folderu src do ścieżki importów
#dzięki temu można zaimportować klasę modelu z pliku src/model.py
sys.path.append("/home/site/wwwroot/src")

from model import BloodCellCNN


#utworzenie aplikacji Azure Functions
app = func.FunctionApp()


#ścieżka do checkpointa zapakowanego w obrazie Docker
CHECKPOINT_PATH = "/home/site/wwwroot/checkpoints/best.ckpt"


#nazwy klas w takiej samej kolejności jak w BloodMNIST
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


#transformacje obrazu takie same jak przy trenowaniu modelu
#obraz jest zmniejszany do 28x28 i normalizowany
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


#funkcja serverless będzie działać na cpu
device = torch.device("cpu")


#wczytanie wytrenowanego modelu z checkpointa
model = BloodCellCNN.load_from_checkpoint(
    CHECKPOINT_PATH,
    map_location=device,
)


#przełączenie modelu w tryb predykcji
model.eval()
model.to(device)


#nazwa funkcji widoczna w Azure
@app.function_name(name="PredictBloodCell")

#trigger uruchamia funkcję po dodaniu pliku do kontenera input-images
@app.blob_trigger(
    arg_name="input_blob",
    path="input-images/{name}",
    connection="AzureWebJobsStorage",
)
def predict_blood_cell(input_blob: func.InputStream):
    #odczytanie bajtów obrazu z Blob Storage
    image_bytes = input_blob.read()

    #zamiana bajtów na obraz RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    #przygotowanie obrazu do formatu oczekiwanego przez model
    image_tensor = transform(image).unsqueeze(0).to(device)

    #wykonanie predykcji bez liczenia gradientów
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class_id = int(torch.argmax(probabilities).item())

    #przygotowanie wyniku predykcji do zapisania jako json
    result = {
        "source_blob": input_blob.name,
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": CLASS_NAMES[predicted_class_id],
        "probabilities": probabilities.cpu().tolist(),
    }

    #pobranie connection stringa do Storage Account z ustawień Azure Function
    connection_string = os.environ["AzureWebJobsStorage"]

    #utworzenie klienta do komunikacji z Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    #przygotowanie nazwy pliku wynikowego
    input_file_name = Path(input_blob.name).name
    result_file_name = f"{Path(input_file_name).stem}.json"

    #wskazanie miejsca zapisu wyniku w kontenerze results
    result_blob_client = blob_service_client.get_blob_client(
        container="results",
        blob=result_file_name,
    )

    #zapisanie wyniku predykcji do Blob Storage
    result_blob_client.upload_blob(
        json.dumps(result, indent=2),
        overwrite=True,
    )