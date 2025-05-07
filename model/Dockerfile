# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# Copy your FastAPI app
COPY . .

# Expose FastAPI port
EXPOSE 8001

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8001"]
