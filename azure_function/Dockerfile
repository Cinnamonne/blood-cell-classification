FROM mcr.microsoft.com/azure-functions/python:4-python3.11

WORKDIR /home/site/wwwroot

COPY azure_function/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY azure_function/function_app.py .
COPY azure_function/host.json .

COPY src ./src
COPY checkpoints ./checkpoints