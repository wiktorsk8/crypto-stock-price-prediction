FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

# Run the application when the container starts
CMD ["python", "src/main.py"]
