from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_fscore_support,
)


def score_medio_vp_vn(y_true, y_pred) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    vn, fp, fn, vp = cm.ravel()

    sensibilidade = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0.0

    return (sensibilidade + especificidade) / 2


def avaliar_modelo(modelo, X_test, y_test, nome: str = "Modelo") -> dict:
    y_pred = modelo.predict(X_test)

    # métricas agregadas (visão geral)
    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # métricas por classe (0 = legítimo, 1 = golpe/phishing)
    precisao_classe, recall_classe, f1_classe, suporte_classe = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    resultados["precision_legitimo"] = precisao_classe[0]
    resultados["recall_legitimo"] = recall_classe[0]
    resultados["f1_legitimo"] = f1_classe[0]
    resultados["suporte_legitimo"] = int(suporte_classe[0])

    resultados["precision_golpe"] = precisao_classe[1]
    resultados["recall_golpe"] = recall_classe[1]
    resultados["f1_golpe"] = f1_classe[1]
    resultados["suporte_golpe"] = int(suporte_classe[1])

    # nova métrica: média entre sensibilidade (VP) e especificidade (VN)
    resultados["score_medio_vp_vn"] = score_medio_vp_vn(y_test, y_pred)

    print(f"--- Resultados: {nome} ---")
    print(f"Accuracy: {resultados['accuracy']:.4f}")
    print(f"Precision (weighted): {resultados['precision_weighted']:.4f}")
    print(f"Recall (weighted): {resultados['recall_weighted']:.4f}")
    print(f"F1-score (weighted): {resultados['f1_weighted']:.4f}")
    print(f"Score médio VP/VN (sensibilidade+especificidade / 2): {resultados['score_medio_vp_vn']:.4f}")

    print(f"\n--- Métricas por classe ---")
    print(f"{'Classe':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Suporte':>12}")
    print(f"{'Legítimo (0)':<12}{resultados['precision_legitimo']:>12.4f}"
          f"{resultados['recall_legitimo']:>12.4f}{resultados['f1_legitimo']:>12.4f}"
          f"{resultados['suporte_legitimo']:>12d}")
    print(f"{'Golpe (1)':<12}{resultados['precision_golpe']:>12.4f}"
          f"{resultados['recall_golpe']:>12.4f}{resultados['f1_golpe']:>12.4f}"
          f"{resultados['suporte_golpe']:>12d}")

    print(f"\n>> Recall na classe 'golpe' (taxa de detecção real): {resultados['recall_golpe']:.4f}")
    print(f">> Precision na classe 'legítimo' (confiabilidade dos negativos): {resultados['precision_legitimo']:.4f}")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred, target_names=["legitimo", "golpe"], zero_division=0))

    return resultados


def plotar_matriz_confusao(modelo, X_test, y_test, nome: str = "Modelo", salvar_em: str = None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["legitimo", "golpe"], yticklabels=["legitimo", "golpe"])
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Predito")
    plt.ylabel("Real")

    if salvar_em:
        plt.savefig(salvar_em, bbox_inches="tight")
        print(f"Gráfico salvo em {salvar_em}")
    else:
        plt.show()

    plt.close()