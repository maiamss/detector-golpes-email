import argparse
import json
import os

import joblib
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.preprocess import preprocess_batch, build_feature_matrix
from ml.metrics import avaliar_modelo, score_medio_vp_vn
from ml.logging_utils import configurar_logger, registrar_experimento
from ml.train import carregar_dados, preparar_dados, BASE_DIR, MODELS_DIR

# deixa os logs do próprio Optuna mais enxutos no console
optuna.logging.set_verbosity(optuna.logging.WARNING)

LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

N_SPLITS_CV = 5
RANDOM_STATE = 42

logger = configurar_logger("optuna_tuning")


def avaliar_via_cv(modelo, X, y, cv) -> float:
    scores = []
    for idx_treino, idx_val in cv.split(X, y):
        X_tr, X_val = X[idx_treino], X[idx_val]
        y_tr, y_val = y.iloc[idx_treino], y.iloc[idx_val]

        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_val)
        scores.append(score_medio_vp_vn(y_val, y_pred))

    return sum(scores) / len(scores)

def objective_logreg(trial, X, y, cv):
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    modelo = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return avaliar_via_cv(modelo, X, y, cv)


def objective_rf(trial, X, y, cv):
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 40)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return avaliar_via_cv(modelo, X, y, cv)


def objective_mlp(trial, X, y, cv):
    hidden_layer_sizes = trial.suggest_categorical(
        "hidden_layer_sizes",
        [(50,), (100,), (100, 50), (150, 100), (200, 100, 50)],
    )
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])

    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        activation=activation,
        solver=solver,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    return avaliar_via_cv(modelo, X, y, cv)


# construtor do modelo "final" a partir dos best_params do Optuna
CONSTRUTORES_FINAIS = {
    "logreg": lambda p: LogisticRegression(
        **p, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "rf": lambda p: RandomForestClassifier(
        **p, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "mlp": lambda p: MLPClassifier(
        **p, max_iter=300, early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE
    ),
}

OBJECTIVES = {
    "logreg": objective_logreg,
    "rf": objective_rf,
    "mlp": objective_mlp,
}


def rodar_estudo(nome_modelo: str, X_train, y_train, X_test, y_test, n_trials: int):
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    objective_fn = OBJECTIVES[nome_modelo]

    logger.info(f"=== Optuna: otimizando {nome_modelo} ({n_trials} trials) ===")

    study = optuna.create_study(
        study_name=f"tuning_{nome_modelo}",
        direction="maximize",  # queremos MAXIMIZAR o score_medio_vp_vn
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv),
        n_trials=n_trials,
        n_jobs=1,  # cuidado ao paralelizar junto com n_jobs=-1 do RF/MLP
        show_progress_bar=True,
    )

    logger.info(f"Melhor score (CV) para {nome_modelo}: {study.best_value:.4f}")
    logger.info(f"Melhores hiperparâmetros: {study.best_params}")

    # salva o histórico completo de trials (bom material pra um gráfico no TCC:
    # "score vs. número de trials", ou tabela dos top-5 configs testadas)
    caminho_trials = os.path.join(LOGS_DIR, f"optuna_trials_{nome_modelo}.csv")
    study.trials_dataframe().to_csv(caminho_trials, index=False)
    logger.info(f"Histórico de trials salvo em {caminho_trials}")

    # treina o modelo FINAL com os melhores hiperparâmetros, no treino inteiro,
    # e só agora avalia no conjunto de teste (que o Optuna nunca viu)
    modelo_final = CONSTRUTORES_FINAIS[nome_modelo](study.best_params)
    modelo_final.fit(X_train, y_train)
    resultados = avaliar_modelo(modelo_final, X_test, y_test, nome=f"{nome_modelo} (Optuna)")

    joblib.dump(modelo_final, os.path.join(MODELS_DIR, f"modelo_{nome_modelo}_optuna.pkl"))

    with open(os.path.join(LOGS_DIR, f"best_params_{nome_modelo}.json"), "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)

    registrar_experimento({
        "modelo": f"{nome_modelo}_optuna",
        "hiperparametros": str(study.best_params),
        "score_cv": study.best_value,
        **{k: v for k, v in resultados.items()},
    })

    return study, modelo_final, resultados


def main():
    parser = argparse.ArgumentParser(description="Tuning de hiperparâmetros com Optuna")
    parser.add_argument(
        "--modelo", choices=["logreg", "rf", "mlp", "todos"], default="todos",
        help="Qual modelo otimizar (default: todos)",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Número de tentativas do Optuna por modelo (default: 30)",
    )
    args = parser.parse_args()

    df = carregar_dados()
    df = preparar_dados(df)
    y = df["label"]

    idx_train, idx_test, y_train, y_test = train_test_split(
        df.index, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    raw_train = df.loc[idx_train, "text_combined"].tolist()
    raw_test = df.loc[idx_test, "text_combined"].tolist()
    clean_train = df.loc[idx_train, "text_clean"].tolist()
    clean_test = df.loc[idx_test, "text_clean"].tolist()

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    scaler = MinMaxScaler()

    X_train = build_feature_matrix(raw_train, clean_train, vectorizer, scaler, fit=True)
    X_test = build_feature_matrix(raw_test, clean_test, vectorizer, scaler, fit=False)

    modelos = ["logreg", "rf", "mlp"] if args.modelo == "todos" else [args.modelo]

    for nome_modelo in modelos:
        rodar_estudo(nome_modelo, X_train, y_train, X_test, y_test, n_trials=args.trials)

    logger.info("Tuning concluído. Veja logs/historico_experimentos.csv para comparar tudo.")


if __name__ == "__main__":
    main()