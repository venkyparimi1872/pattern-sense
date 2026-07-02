FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
