FROM python:3.12.6

WORKDIR /app

# Instalar ferramentas adicionais, bibliotecas e o Xvfb
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    vim \
    xvfb  
# Adicionando Xvfb

# Copiar o arquivo de requisitos e instalar as dependências de Python
COPY requirements.txt ./
COPY app/ ./app/
COPY assets/ ./assets/
COPY output/ ./output/

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Usando uma lista correta para o CMD
CMD ["xvfb-run", "-a", "python", "app/canny_improved.py"]
