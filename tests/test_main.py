from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analisar_email_golpe():
    response = client.post(
        "/analisar-email",
        json={
            "email_text": (
                "URGENT: Your bank account has been suspended due to suspicious activity. "
                "Verify your account immediately by clicking here or it will be closed within 24 hours."
            )
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["risco"] == "alto"
    assert data["classificacao"] == "golpe"
    assert data["score"] >= 0.6
    assert len(data["explicacao"]) > 0


def test_analisar_email_legitimo():
    response = client.post(
        "/analisar-email",
        json={"email_text": "Hi Sarah, thanks for sending the draft. I left a few comments in the shared document. Can we talk tomorrow at 10 to go over them? Best, Mark"},
    )
    assert response.status_code == 200
    assert response.json()["risco"] == "baixo"


def test_email_vazio_retorna_erro():
    # O Pydantic (min_length=1 no schema) barra a string vazia antes de
    # chegar no endpoint, retornando 422 (erro de validação padrão do FastAPI).
    response = client.post("/analisar-email", json={"email_text": ""})
    assert response.status_code == 422
