# 1. Usa uma imagem oficial do Python como base
FROM python:3.11-slim

# 2. Define a pasta de trabalho dentro do contêiner
WORKDIR /app

# 3. Copia o arquivo de dependências para o contêiner
COPY requirements.txt .

# 4. Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia todo o resto do seu código para dentro do contêiner
COPY . .

# 6. Define o comando padrão para executar seu script
CMD ["python", "main.py"]
