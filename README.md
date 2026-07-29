# Backend — Sistema de Detecção de Golpes Bancários via E-mail

Backend do TCC **"Sistema Web para Detecção e Alerta de Golpes Bancários via
E-mail utilizando Processamento de Linguagem Natural e Classificação
Supervisionada"**.

Responsável por: receber o texto do e-mail (enviado pela extensão do
navegador), aplicar o modelo de classificação treinado e retornar o nível de
risco com explicação dos padrões detectados.

---

## Arquitetura (visão geral do TCC)

```
Usuário → extensão de navegador → [requisição HTTP] → API (este backend)
                                                            │
                                                            ▼
                                                  Modelo de IA
                                     (TF-IDF+LogReg / SVM / DistilBERT)
```

Este repositório cobre apenas a caixa **API (FastAPI)** do fluxograma. A
extração do texto do e-mail é responsabilidade da extensão (frontend); o
treinamento do modelo é responsabilidade de outra pessoa do grupo.

---

## Estrutura de pastas

```
backend/
├── app/
│   ├── main.py                 # Cria a API, registra rotas, configura CORS
│   ├── core/
│   │   └── config.py           # Variáveis de ambiente e caminhos de modelo
│   ├── models/
│   │   └── schemas.py          # Contrato JSON (Pydantic) de request/response
│   └── services/
│       └── classifier.py       # Lógica de classificação (hoje: mock)
├── model_files/                # (a criar) onde entram modelo.pkl e vectorizer.pkl
├── tests/
│   └── test_main.py            # Testes automatizados dos endpoints
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

A API sobe em `http://localhost:8000`.

- Swagger UI (documentação interativa, testável no navegador):
  `http://localhost:8000/docs`
- Redoc (documentação alternativa, somente leitura):
  `http://localhost:8000/redoc`
- Especificação OpenAPI crua (JSON):
  `http://localhost:8000/openapi.json`

O Swagger é gerado **automaticamente pelo FastAPI** a partir dos schemas
Pydantic e das descrições dos endpoints em `main.py` — não precisa escrever
nada manualmente além disso.

### Rodando os testes

```bash
pip install pytest httpx
python -m pytest tests/ -v
```

---

## Estado atual: modelo mockado

O arquivo `app/services/classifier.py` contém uma lógica **provisória**
baseada em busca de palavras-chave (urgência, solicitação de dados, links).
Isso existe só para o backend funcionar de ponta a ponta (extensão ↔ API)
enquanto o modelo de verdade não está pronto.

**Quando o modelo real chegar:**

1. Colocar os arquivos entregues (ex: `modelo.pkl`, `vectorizer.pkl`) na
   pasta `model_files/`.
2. Reescrever a função `classificar_email()` em `classifier.py` para:
   - Carregar o modelo e o vetorizador (uma vez, na inicialização, não a
     cada requisição).
   - Aplicar o mesmo pré-processamento usado no treino.
   - Rodar a predição e montar o `EmailAnaliseResponse`.
3. Manter a assinatura da função (`str` → `EmailAnaliseResponse`) para não
   quebrar o restante da API.

---

## Contrato JSON da API

### `POST /analisar-email`

**Request:**
```json
{
  "email_text": "Prezado cliente, sua conta será bloqueada em 24 horas...",
  "email_subject": "URGENTE: Regularize sua conta",
  "sender": "seguranca@banco-verificacao.com"
}
```

`email_subject` e `sender` são opcionais.

**Response (200):**
```json
{
  "risco": "alto",
  "score": 0.8,
  "classificacao": "golpe",
  "explicacao": [
    {
      "tipo": "urgencia_artificial",
      "descricao": "Uso de expressão de urgência",
      "trecho": "bloqueada"
    }
  ],
  "modelo_usado": "mock_v0",
  "tempo_inferencia_ms": 0.12
}
```

**Response de erro de validação (422):** e-mail vazio ou campo ausente
(gerado automaticamente pelo FastAPI/Pydantic).

### `GET /health`

Retorna o status da API e se o modelo está carregado. Útil para checar se o
deploy está de pé.

```json
{
  "status": "ok",
  "modelo_carregado": true,
  "versao_modelo": "mock_v0"
}
```

---

## Variáveis de ambiente

| Variável           | Padrão                              | Descrição                                   |
|---------------------|--------------------------------------|----------------------------------------------|
| `MODEL_PATH`        | `model_files/modelo.pkl`            | Caminho do modelo treinado                    |
| `VECTORIZER_PATH`   | `model_files/vectorizer.pkl`        | Caminho do vetorizador TF-IDF                 |
| `MODEL_VERSION`     | `mock_v0`                            | Identificador da versão do modelo em uso      |
| `ALLOWED_ORIGINS`   | `*`                                   | Origens permitidas por CORS (ajustar no deploy)|

Para configurar localmente, crie um arquivo `.env` na raiz de `backend/`
(não versionar, adicionar ao `.gitignore`).

---

## Deploy (gratuito)

Sugestão: **Render (free tier)**.

1. Subir o repositório no GitHub.
2. Criar um "Web Service" no Render apontando para a pasta `backend/`.
3. Comando de build: `pip install -r requirements.txt`
4. Comando de start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Configurar as variáveis de ambiente da tabela acima no painel do Render.

⚠️ No plano gratuito o serviço "dorme" após ~15 min sem uso e demora para
"acordar" na primeira requisição seguinte — normal, não é bug.

⚠️ Se o modelo escolhido for **DistilBERT**, validar se o free tier
(RAM/CPU) aguenta o carregamento do modelo antes de depender dele na
apresentação do TCC.

---

## Próximos passos (checklist do backend)

- [ ] Alinhar com quem treina o modelo: formato de entrega do `.pkl`/pesos
- [ ] Trocar o mock em `classifier.py` pelo modelo real
- [ ] Adicionar `model_files/` ao `.gitignore` se os arquivos forem grandes
- [ ] Ajustar `ALLOWED_ORIGINS` para a origem real da extensão antes do deploy
- [ ] Testar o endpoint `/analisar-email` com e-mails reais do dataset
- [ ] Medir `tempo_inferencia_ms` com o modelo real (não só o mock)
