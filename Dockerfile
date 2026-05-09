FROM python:3.10-slim

# Empêcher Python de bufferiser stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit avec les bons paramètres réseau
CMD ["streamlit", "run", "orion_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
