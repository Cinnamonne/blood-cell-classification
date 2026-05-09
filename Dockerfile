FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install torch torchvision
RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["bentoml", "serve", "service:BloodCellClassifier", "--host", "0.0.0.0", "--port", "3000"]