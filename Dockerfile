FROM pytorch/pytorch:latest

WORKDIR /workspace
COPY . .

RUN pip install -r requirements.txt