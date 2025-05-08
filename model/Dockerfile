# Use the official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# Copy your FastAPI app
COPY . .

# Set Hugging Face cache dir
ENV HF_HOME=/app/.cache/huggingface

# Download the model at build time
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct'); \
    AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-360M-Instruct')"

# Expose FastAPI port
EXPOSE 8001

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8001"]
