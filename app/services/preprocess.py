import re #regular expressions -> encontra e remove padrões
import spacy #usada para PLN (tokenização, identificação de palavras e lematização)
from nltk.corpus import stopwords #palavras comuns

#modelo em inglês
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) #carrega o modelo em inglês e desativa a análise sintática + reconhecimento de entidades nomeadas (pessoas, empresas e lugares)
stop_words = set(stopwords.words("english")) #carrega as stopwords em inglês e transforma a lista em conjunto

def preprocess_text(text: str) -> str:
    #caso esteja vazio
    if not isinstance(text, str): #verifica se o conteúdo recebido é uma string
        return ""   
    text = text.lower() #minúsculas
    text = re.sub(r"http\S+|www\S+", "", text) #remove URLs
    text = re.sub(r"\d+", "", text) #remover números
    text = re.sub(r"[^\w\s]", "", text) #remover pontuação

    doc = nlp(text) #texto é limpo e enviado para o spacy
    palavras = [
        token.lemma_ for token in doc
        if token.text not in stop_words and not token.is_space
    ]
    return " ".join(palavras)

def preprocess_batch(texts: list[str], batch_size: int = 100, max_chars: int=20000) -> list[str]: #mesmo objetivo da funça2o anterior, porém trabalha com mais textos de uma vez. Define que irá trabalhar com 100 textos de uma vez
    textos_limpos = []
    for text in texts:
        if not isinstance(text, str):
            textos_limpos.append("")
            continue
        t = text[:max_chars].lower()
        t = re.sub(r"http\S+|www\S+", "", t) 
        t = re.sub(r"\d+", "", t)
        t = re.sub(r"[^\w\s]", "", t)
        textos_limpos.append(t)

    resultados = []
    for doc in nlp.pipe(textos_limpos, batch_size=batch_size): #processa vários textos de uma vez
        palavras = [
            token.lemma_ for token in doc
            if token.text not in stop_words and not token.is_space
        ]
        resultados.append(" ".join(palavras))

    return resultados

def extract_extra_features(text: str) -> dict: #características extras do e-mail
    if not isinstance(text, str):
        text = ""

    total_chars = len(text) if len(text) > 0 else 1 

    return {
        "has_url": int(bool(re.search(r"http\S+|www\S", text))), #procura se o texto tem uma URL
        "num_exclamacoes": text.count("!"),
        "pct_maiusculas": sum(1 for c in text if c.isupper()) / total_chars, #qtd. de maiúsculas / qtd. total de caracteres
        "tamanho_texto": len(text),
    }

if __name__ == "__main__":
    #teste rápido
    exemplo = "URGENT!!! Verify your account NOW at http://fake-bank.com or it will be suspended."
    print("Antes: ", exemplo)
    print("Depois: ", preprocess_text(exemplo))
    print("Features extras: ", extract_extra_features(exemplo))