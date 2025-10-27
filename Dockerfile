FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ /app/src/
COPY models/ /app/models/
RUN ls -l /app/models/  # Debug to verify files