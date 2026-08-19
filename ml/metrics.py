from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,)

def avaliar_modelo(modelo, X_test, y_test, nome: str = "Modelo") -> dict:
    
    y_pred = modelo.predict(X_test) #resposta do modelo

    resultados = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0), #average="weighted -> calcula a métrica considerando o tamanho de cada classe"
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print(f"--- Resultados: {nome} ---")
    print(f"Accuracy: {resultados['accuracy']:.4f}")
    print(f"Precision: {resultados['precision']:.4f}")
    print(f"Recall: {resultados['recall']:.4f}")
    print(f"F1-score: {resultados['f1']:.4f}")
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório completo:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return resultados

def plotar_matriz_confusao(modelo, X_test, y_test, nome: str = "Modelo", salvar_em: str = None): #cria um gráfico visual da matriz de confusão
    import matplotlib.pyplot as plt #cria gráficos
    import seaborn as sns #cria o gráfico da matriz de confusão
    
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Predito")
    plt.ylabel("Real")

    if salvar_em:
        plt.savefig(salvar_em, bbox_inches="tight")
        print(f"Gráfico salvo em {salvar_em}")
    else: 
        plt.show()

    plt.close()