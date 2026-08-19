FROM python:3.11-slim

WORKDIR /app

# 1. Copia o arquivo de dependências para o contêiner
COPY requirements.txt .

# 2. Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# 3. Baixa o modelo de linguagem do spaCy e as stopwords do NLTK
#    usados em app/services/preprocess.py
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader stopwords

# 4. Copia todo o resto do seu código para dentro do contêiner
COPY . .

# 5. Expõe a porta usada pelo uvicorn
EXPOSE 8000

# 6. Sobe a API com uvicorn (app/main.py define o objeto `app`)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]