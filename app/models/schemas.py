"""
Schemas Pydantic - definem o contrato JSON da API.
Qualquer mudança aqui deve ser alinhada com o time (frontend/modelo)
antes de ser aplicada, pois afeta o contrato de comunicação.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class NivelRisco(str, Enum):
    baixo = "baixo"
    medio = "medio"
    alto = "alto"


class EmailAnaliseRequest(BaseModel):
    """Dados enviados pela extensão do navegador."""

    email_text: str = Field(
        ...,
        min_length=1,
        description="Corpo do e-mail extraído pela extensão",
        examples=["Prezado cliente, sua conta será bloqueada em 24 horas..."],
    )
    email_subject: Optional[str] = Field(
        default=None, description="Assunto do e-mail (opcional)"
    )
    sender: Optional[str] = Field(
        default=None, description="Remetente do e-mail (opcional)"
    )


class PadraoDetectado(BaseModel):
    tipo: str = Field(..., examples=["urgencia_artificial"])
    descricao: str
    trecho: Optional[str] = None


class EmailAnaliseResponse(BaseModel):
    """Resposta enviada de volta para a extensão."""

    risco: NivelRisco
    score: float = Field(..., ge=0.0, le=1.0, description="Score de 0 a 1")
    classificacao: str = Field(..., examples=["golpe", "legitimo"])
    explicacao: List[PadraoDetectado] = []
    modelo_usado: str
    tempo_inferencia_ms: float


class ErroResponse(BaseModel):
    erro: bool = True
    codigo: str
    mensagem: str


class HealthResponse(BaseModel):
    status: str = "ok"
    modelo_carregado: bool
    versao_modelo: Optional[str] = None

import joblib, numpy as np
from app.services.preprocess import preprocess_text   

vec = joblib.load("app/models/tfidf.pkl")             
clf = joblib.load("app/models/modelo_logistico.pkl")   

for texto in ["Shall we meet for coffee tomorrow?",
              "Hi, how are you? Shall we meet for coffee tomorrow? I was thinking about the place near the park, around three in the afternoon, if that works for you."]:
    x = vec.transform([preprocess_text(texto)])
    nomes = np.array(vec.get_feature_names_out())
    contrib = x.toarray()[0] * clf.coef_[0][:x.shape[1]]
    idx = np.argsort(-np.abs(contrib))[:8]
    print(texto[:40], "| intercepto:", clf.intercept_[0].round(3))
    for i in idx:
        if contrib[i] != 0:
            print(f"  {nomes[i]:<20} {contrib[i]:+.3f}")