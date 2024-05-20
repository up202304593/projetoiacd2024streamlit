import streamlit as st
import pandas as pd
from PIL import Image


header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
    st.title("IACD Project: Data exploration and enrichment for supervised classification ")
    st.text("In this project we adress data science in a more pratical way,\nutilizing the HCC datset to train our data processing, our method of data modeling\nand some forms of data evaluation.The final objetive is to develop a machine learning\n pipeline that determines the survivability of patients 1 year after the diagnosis.")

with dataset:
    st.header("The Hepatocellular Carcinoma Dataset")
    st.text("This dataset provides us with a variety of symptoms,habits and personal information \nof various patients and in the end the ultimate goal of if a patient lived or died. ")
    
    data = pd.read_csv("hcc_dataset.csv")
    st.write(data.head())

    st.text("We can take important symptoms or information as more crucial data that weights\n significantly more than others in the final decision such as the nºLeucocytes: ")
    
    st.subheader("Number of Leucocytes")
    num_leucocitos = pd.DataFrame(data['Leucocytes'].value_counts())
    st.bar_chart(num_leucocitos)

##############################################################################################################
code_snipet1 ="""Percorre cada linha do arquivo original
    for row in rows:
        new_row = []
        for value in row:
            if value == '?':
                new_row.append('Nan')
            elif value == 'Yes':
                new_row.append('1')
            elif value == 'No':
                new_row.append('0')
            else:
                new_row.append(value)
        writer.writerow(new_row)"""
code_snipet2 = """def plot_histograms(data):
    # Definir o número máximo de colunas a serem plotadas de uma vez
    max_columns_per_plot = 3
 
    # Definir o esquema de cores
    color_scheme = ['cadetblue', 'navajowhite', 'aquamarine', 'indianred', 'mediumpurple', 'peru', 'lightpink', 'silver', 'y', 'lightseagreen']

    num_plots = len(data.columns)
    num_rows = math.ceil(num_plots / max_columns_per_plot)

    fig, axes = plt.subplots(num_rows, max_columns_per_plot, figsize=(15, 5*num_rows))
    axes = axes.flatten()

    for i, column in enumerate(data.columns):
        subset_data = data[column]
        ax = axes[i]

        if pd.api.types.is_numeric_dtype(subset_data):
            cleaned_data = subset_data.fillna('None')
            num_bins = int(len(cleaned_data) ** 0.5)
            num_bins = max(10, num_bins)

            if cleaned_data.nunique() > 20:
                num_bins = min(50, cleaned_data.nunique())

            ax.hist(cleaned_data, bins=num_bins, color=color_scheme[i % len(color_scheme)], edgecolor='black')

            # Adicionar barra para 'NaN' e 'None'
            nan_count = subset_data.isna().sum()
            if nan_count > 0:
                ax.bar('NaN', nan_count, color='gray', alpha=0.5)
            
            none_count = (subset_data == 'None').sum()
            if none_count > 0:
                ax.bar('None', none_count, color='gray', alpha=0.5)

        else:
            cleaned_data = subset_data.fillna('None')

            if cleaned_data.nunique() > 15:
                cleaned_data = pd.to_numeric(cleaned_data, errors='coerce')
                cleaned_data = cleaned_data.dropna()
                interval_size = (cleaned_data.max() - cleaned_data.min()) / 10
                bins = [cleaned_data.min() + i * interval_size for i in range(11)]
            else:
                bins = cleaned_data.nunique()

            ax.hist(cleaned_data, bins=bins, color=color_scheme[i % len(color_scheme)], edgecolor='black')

            # Adicionar barra para 'None'
            none_count = (subset_data == 'None').sum()
            if none_count > 0:
                ax.bar('None', none_count, color='gray', alpha=0.5)

        ax.set_title(f'Distribuição de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Número de Pacientes')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# Carregar os dados do arquivo CSV para um DataFrame
data = pd.read_csv("hcc_dataset_modified.csv")

# Criar os gráficos para cada variável
plot_histograms(data)"""

code_snipet3 = """# Verificar o nome exato da última coluna
class_column = data.columns[-1]

# Converter 'lives' para 1 e 'dies' para 0 na coluna 'class'
data[class_column] = data[class_column].replace({'Lives': 1, 'Dies': 0})

# Identificar colunas que podem ser convertidas para numéricas, incluindo aquelas com NaN
numeric_data = data.apply(pd.to_numeric, errors='coerce')

# Calcular a matriz de correlação
correlation_matrix = numeric_data.corr()

# Plotar o heatmap da matriz de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matriz de Correlação")
plt.show()"""

code_snipet4 = """# Verificar o nome exato da última coluna
class_column = data.columns[-1]

# Converter 'lives' para 1 e 'dies' para 0 na coluna 'class'
data[class_column] = data[class_column].replace({'Lives': 1, 'Dies': 0})

# Identificar colunas que podem ser convertidas para numéricas, incluindo aquelas com NaN
numeric_data = data.apply(pd.to_numeric, errors='coerce')

# Mostrar a matriz de correlação
print("Matriz de Correlação:")
print(correlation_matrix)"""

code_snipet5 = """# Calcular a quantidade de vezes que a palavra "Nan" aparece em cada coluna
nan_counts = (data.astype(str) == "Nan").sum()

# Selecionar as colunas com menos ocorrências da palavra "Nan" (menos de 10% dos dados)
relevant_columns_nan = nan_counts[nan_counts < 0.1 * len(data)]

# Selecionar apenas colunas numéricas
numeric_columns = data.select_dtypes(include=[float, int])

# Calcular a variância de cada coluna numérica
variance = numeric_columns.var()

# Selecionar as colunas com maior variância
relevant_columns_variance = variance.nlargest(len(numeric_columns.columns))

# Mostrar as colunas mais relevantes com base na quantidade de vezes que a palavra "Nan" aparece e na variância
print("Colunas mais relevantes com base na quantidade de vezes que a palavra 'Nan' aparece:")
print(relevant_columns_nan)
print("\nColunas mais relevantes com base na variância:")
print(relevant_columns_variance)

# Agora, vamos encontrar as colunas menos relevantes com base na quantidade de vezes que a palavra "Nan" aparece e na variância
print("\nColunas menos relevantes com base na quantidade de vezes que a palavra 'Nan' aparece:")
least_relevant_columns_nan = nan_counts[nan_counts >= 0.1 * len(data)]
print(least_relevant_columns_nan)

print("\nColunas menos relevantes com base na variância:")
least_relevant_columns_variance = variance.nsmallest(len(numeric_columns.columns))
print(least_relevant_columns_variance)
"""
code_snipet6 = """percentagem_nan = {}

# Obter o número total de linhas no DataFrame
total_linhas = len(data)

# Percorrer as colunas do DataFrame
for coluna in data.columns:
    # Contar o número de ocorrências de 'Nan' na coluna atual
    contagem_nan = (data[coluna] == 'Nan').sum()
    
    # Calcular a percentagem de 'Nan' na coluna atual
    percentagem = (contagem_nan / total_linhas) * 100
    
    # Armazenar a percentagem de 'Nan' na coluna atual no dicionário
    percentagem_nan[coluna] = percentagem

# Exibir a percentagem de 'Nan' em cada categoria
for categoria, percentagem in percentagem_nan.items():
    print(f"{categoria}: {percentagem:.2f}%")"""

code_snipet7 = """import pandas as pd

# Função para remover colunas do DataFrame
def remove_columns(data, columns_to_drop):
    columns = [col[0] for col in columns_to_drop]
    return data.drop(columns=columns)

# Carregar os dados do arquivo CSV para um DataFrame
data = pd.read_csv("hcc_dataset_modified.csv")

# Verificar o nome exato da última coluna
class_column = data.columns[-1]

# Converter 'Lives' para 1 e 'Dies' para 0 na coluna 'class'
data[class_column] = data[class_column].replace({'Lives': 1, 'Dies': 0})

# Identificar colunas que podem ser convertidas para numéricas, incluindo aquelas com NaN
numeric_data = data.apply(pd.to_numeric, errors='coerce')

# Calculando as correlações entre as colunas
correlation_matrix = numeric_data.corr()

# Encontrar todos os pares de colunas com correlação superior a 0.6 entre si (exceto a última coluna)
highly_correlated_pairs = []
for i, col1 in enumerate(correlation_matrix.columns[:-1]):
    for col2 in correlation_matrix.columns[i + 1:-1]:
        corr = correlation_matrix.loc[col1, col2]
        if abs(corr) > 0.6:
            highly_correlated_pairs.append((col1, col2, corr))

# Calcular a correlação entre cada coluna e a última coluna
correlation_with_class = correlation_matrix.iloc[:-1, -1]

# Escolher a coluna mais correlacionada com a última coluna para cada par de colunas
columns_to_keep = []
columns_to_drop = []
for pair in highly_correlated_pairs:
    col1, col2, corr_pair = pair
    corr_col1 = correlation_with_class[col1]
    corr_col2 = correlation_with_class[col2]
    # Comparar as correlações considerando a direção dos números negativos
    if corr_col1 >= 0 and corr_col2 >= 0:
        column_to_keep = col1 if corr_col1 > corr_col2 else col2
        column_to_drop = col2 if corr_col1 > corr_col2 else col1
    elif corr_col1 < 0 and corr_col2 < 0:
        column_to_keep = col1 if corr_col1 > corr_col2 else col2
        column_to_drop = col2 if corr_col1 > corr_col2 else col1
    else:
        column_to_keep = col1 if corr_col1 >= corr_col2 else col2
        column_to_drop = col2 if corr_col1 >= corr_col2 else col1
    columns_to_keep.append(column_to_keep)
    columns_to_drop.append((column_to_drop, corr_col1, corr_col2, corr_pair))

# Remover as colunas menos correlacionadas com a última coluna
data = remove_columns(data, columns_to_drop)

# Salvar o DataFrame atualizado em um novo arquivo
# Substitua 'caminho_para_o_novo_arquivo.csv' pelo caminho do novo arquivo
data.to_csv('hcc_dataset_modified_2.csv', index=False, na_rep='None')

# Imprimindo as colunas a serem mantidas
print("Colunas a serem mantidas:")
for col in columns_to_keep:
    print(f"  {col}")

# Imprimindo as colunas a serem eliminadas e suas correlações
print("\nColunas a serem eliminadas e suas correlações:")
for col, corr_col1, corr_col2, corr_pair in columns_to_drop:
    print(f"  {col}:")
    print(f"    - Correlação com a última coluna: {corr_col1}")
    print(f"    - Correlação com a última coluna: {corr_col2}")
    print(f"    - Correlação entre si: {corr_pair}")

print(data.head())"""

code_snipet8 = """from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def load_data(file_path):
    Carrega o conjunto de dados.
    dataset = pd.read_csv(file_path)
    return dataset

# Carregar o conjunto de dados
file_path = "hcc_dataset_completo.csv"
dataset = load_data(file_path)

# Definir as colunas de características (X) e a coluna de destino (Y)
X = dataset.drop(columns=['Class', 'Gender', 'PS', 'Encephalopathy', 'Ascites'])
Y = dataset['Class']

# Definir as técnicas de balanceamento de dados
data_balancing_techniques = {
    'Original': None,  # Sem balanceamento
    'Down-sampling': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    # Adicionar mais técnicas conforme necessário
}

# Treinar e avaliar modelos para cada técnica de balanceamento
for column in X.columns:
    print(f"--- Avaliação para a coluna '{column}' ---")
    for technique_name, sampler in data_balancing_techniques.items():
        if sampler is not None:
            # Aplicar a técnica de balanceamento de dados
            X_balanced, Y_balanced = sampler.fit_resample(X[[column]].astype(np.float64), Y)
        else:
            # Usar o conjunto de dados original
            X_balanced, Y_balanced = X[[column]].astype(np.float64), Y
        
        # Treinar o modelo (usaremos uma Decision Tree para este exemplo)
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_balanced, Y_balanced)
        
        # Fazer previsões
        y_pred = classifier.predict(X[[column]])
        
        # Avaliar o modelo
        accuracy = accuracy_score(Y, y_pred)
        precision = precision_score(Y, y_pred)
        recall = recall_score(Y, y_pred)
        f1 = f1_score(Y, y_pred)
        
        # Exibir os resultados da avaliação
        print(f"Avaliação para {technique_name}:")
        print(f"   Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    print()"""
code_snipet9 = """# Carregar o conjunto de dados
dataset = pd.read_csv("hcc_dataset_completo.csv")

# Identificar colunas categóricas
categorical_columns = ['Gender', 'PS', 'Encephalopathy', 'Ascites']

# Aplicar Label Encoding para colunas categóricas
label_encoder = LabelEncoder()
for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Separar as variáveis independentes (X) e a variável dependente (Y)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Dividir os dados em conjuntos de treino e teste (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o classificador Decision Tree
classifier_dt = DecisionTreeClassifier(random_state=42)  # Garantir reproducibilidade
classifier_dt.fit(X_train, Y_train)

# Fazer previsões no conjunto de teste
y_pred_dt = classifier_dt.predict(X_test)

# Avaliar o modelo
print("Classification Report for Decision Tree:")
print(classification_report(Y_test, y_pred_dt))
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(Y_test, y_pred_dt))

# Avaliação usando validação cruzada
cv_scores_dt = cross_val_score(classifier_dt, X, Y, cv=10)
print(f"Acurácia média com validação cruzada: {cv_scores_dt.mean()}")

# Calcular a Curva ROC e a Área sob a Curva (AUC) para Decision Tree
y_probs_dt = classifier_dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(Y_test, y_probs_dt)
auc_dt = roc_auc_score(Y_test, y_probs_dt)

# Plotar a Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='orange', label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plotar Matriz de Confusão
conf_matrix_dt = confusion_matrix(Y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plotar Gráfico de Barras para Acurácias da Validação Cruzada
plt.figure(figsize=(8, 6))
plt.bar(np.arange(1, 11), cv_scores_dt, color='skyblue')
plt.title("Acurácias da Validação Cruzada")
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.tight_layout()
plt.show()"""

code_snipet10 = """def plot_diffs(data, column1, column2):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=column1, y=column2, data=data)
    plt.title(f"Comparação entre {column1} e {column2}")
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.show()

# Exemplo de uso:
plot_diffs(data, 'Class', 'Alcohol')" """

code_snipet11 = """# Carregar o dataset
dataset = pd.read_csv("hcc_dataset_completo.csv")

# Identificar colunas categóricas (neste caso, 'Gender', 'PS', etc.)
categorical_columns = ['Gender', 'PS', 'Encephalopathy', 'Ascites']

# Aplicar Label Encoding para colunas categóricas
label_encoder = LabelEncoder()
for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])

# Separar as variáveis independentes (X) e a variável dependente (Y)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Dividir os dados em conjuntos de treino e teste (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ajustar os hiperparâmetros do KNN usando GridSearchCV
param_grid = {'n_neighbors': np.arange(1, min(31, len(X_train)))}  # Limiting the range of n_neighbors
knn = KNeighborsClassifier()

knn_gscv = GridSearchCV(knn, param_grid, cv=4) # cv = 4, 6, 10, 12 -> accuracy = 0.77 -> melhor
knn_gscv.fit(X_train, Y_train)

# Melhor número de vizinhos
print(f"Melhor número de vizinhos para KNN: {knn_gscv.best_params_['n_neighbors']}")

# Treinar o classificador KNN com os melhores parâmetros
classifier_knn = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
classifier_knn.fit(X_train, Y_train)

# Fazer previsões no conjunto de teste
y_pred_knn = classifier_knn.predict(X_test)

# Avaliar o modelo
print("Classification Report for KNN:")
print(classification_report(Y_test, y_pred_knn))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test, y_pred_knn))

# Avaliação usando validação cruzada para KNN
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_knn = cross_val_score(classifier_knn, X, Y, cv=cv)
print(f"Acurácia média com validação cruzada para KNN: {cv_scores_knn.mean()}")

# Calcular a Curva ROC e a Área sob a Curva (AUC) para KNN
y_probs_knn = classifier_knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(Y_test, y_probs_knn)
auc_knn = roc_auc_score(Y_test, y_probs_knn)

# Plotar a Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='green', label=f'KNN (AUC = {auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN')
plt.legend(loc='lower right')
plt.show()

# Plotar Matriz de Confusão
conf_matrix = confusion_matrix(Y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix for KNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plotar Gráfico de Barras para Acurácias da Validação Cruzada
plt.figure(figsize=(12, 4))
plt.bar(np.arange(1, 11), cv_scores_knn, color='green')
plt.title("Cross-Validation Accuracies for KNN")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()"""

code_snipet12 = """# Treinar o classificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)  # Garantir reproducibilidade
rf_classifier.fit(X_train, Y_train)

# Fazer previsões no conjunto de teste
y_pred_rf = rf_classifier.predict(X_test)

# Avaliar o modelo
print("Classification Report for Random Forest:")
print(classification_report(Y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test, y_pred_rf))

# Avaliação usando validação cruzada
cv_scores_rf = cross_val_score(rf_classifier, X, Y, cv=10)
print(f"Acurácia média com validação cruzada (Random Forest): {cv_scores_rf.mean()}")

# Calcular a Curva ROC e a Área sob a Curva (AUC) para Random Forest
y_probs_rf = rf_classifier.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_probs_rf)
auc_rf = roc_auc_score(Y_test, y_probs_rf)

# Plotar a Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc='lower right')
plt.show()

# Plotar Matriz de Confusão
conf_matrix_rf = confusion_matrix(Y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão (Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Plotar Gráfico de Barras para Acurácias da Validação Cruzada
plt.figure(figsize=(8, 6))
plt.bar(np.arange(1, 11), cv_scores_rf, color='skyblue')
plt.title("Acurácias da Validação Cruzada (Random Forest)")
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.tight_layout()
plt.show()"""

code_snipet13 = """# Treinar o classificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, Y_train)

# Fazer previsões no conjunto de teste para Random Forest
y_pred_rf = rf_classifier.predict(X_test)

# Imprimir a tabela para Decision Tree
print("Classification Report for Decision Tree:")
print(classification_report(Y_test, y_pred_dt))
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(Y_test, y_pred_dt))

# Imprimir a tabela para KNN
print("Classification Report for KNN:")
print(classification_report(Y_test, y_pred_knn))
print("Confusion Matrix for KNN:")
print(confusion_matrix(Y_test, y_pred_knn))

# Imprimir a tabela para Random Forest
print("Classification Report for Random Forest:")
print(classification_report(Y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(Y_test, y_pred_rf))


# Avaliação usando validação cruzada para Random Forest
cv_scores_rf = cross_val_score(rf_classifier, X, Y, cv=cv)
print(f"Acurácia média com validação cruzada para Random Forest: {cv_scores_rf.mean()}")

# Calcular a Curva ROC e a Área sob a Curva (AUC) para Random Forest
y_probs_rf = rf_classifier.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_probs_rf)
auc_rf = roc_auc_score(Y_test, y_probs_rf)

# Comparação de resultados
rf_classification_report = classification_report(Y_test, y_pred_rf, output_dict=True)

# Criar DataFrame para métricas
metrics_data = {
    'Metric': metrics,
    'Decision Tree': [dt_classification_report['weighted avg'][metric] for metric in metrics],
    'KNN': [knn_classification_report['weighted avg'][metric] for metric in metrics],
    'Random Forest': [rf_classification_report['weighted avg'][metric] for metric in metrics]  # Adicionando Random Forest
}

metrics_df = pd.DataFrame(metrics_data)

# Adicionar linha para AUCs
metrics_df.loc[len(metrics_df)] = ['AUC', auc_dt, auc_knn, auc_rf]

print("Comparison of Classification Metrics:")
print(metrics_df)


# Plotar as curvas ROC dos três classificadores em um único gráfico
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='orange', label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot(fpr_knn, tpr_knn, color='green', label=f'KNN (AUC = {auc_knn:.2f})')
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Comparison')
plt.legend(loc='lower right')
plt.show()

# Plotar Matriz de Confusão para Random Forest
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(Y_test, y_pred_dt), annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix for Decision Tree")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Plotar Matriz de Confusão para KNN
plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix(Y_test, y_pred_knn), annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix for KNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Plotar Matriz de Confusão para Random Forest
plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix(Y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# Plotar Gráfico de Barras para Acurácias da Validação Cruzada
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(np.arange(1, 11), cv_scores_dt, color='orange')
plt.title("Cross-Validation Accuracies for Decision Tree")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 2)
plt.bar(np.arange(1, 11), cv_scores_knn, color='green')
plt.title("Cross-Validation Accuracies for KNN")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.subplot(1, 3, 3)
plt.bar(np.arange(1, 11), cv_scores_rf, color='blue')
plt.title("Cross-Validation Accuracies for Random Forest")
plt.xlabel("Fold")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()"""

################################################################################################################

with features:
    st.header("The pre-processing of the data")

    st.markdown("* **Values Substitution:**")
    with st.expander("Substitution Code"):
        st.code(code_snipet1, language='python')


    st.markdown("* **Graphics for each Info:**")
    with st.expander("Graphics Code"):
        st.code(code_snipet2, language='python')
        st.image("graphinfo.png", caption = "Image of Graphics", use_column_width=True)
    
    
    st.markdown("* **Correlation Matrix:**")
    with st.expander("Matrix Code"):
        st.code(code_snipet3, language='python')
        st.image("correlationmatrix.png", caption = "Correlation Matrix", use_column_width=True)


    st.markdown("* **Numerical Matrix:**")
    with st.expander("Num Matrix Code"):
        st.code(code_snipet4, language='python')

    st.markdown("* **Relevant Columns:**")
    with st.expander("Relevant Columns Code"):
        st.code(code_snipet5, language='python')

    st.markdown("* **% Nan:**")
    with st.expander("%Nan Code"):
        st.code(code_snipet6, language='python')

    st.markdown("* **Remoção de Colunas Não-Impactantes:**")
    with st.expander("Rem Col Code"):
        st.code(code_snipet7, language='python')

    st.markdown("* **Plot_diff**")
    with st.expander("Plot_Diff Code"):
        st.text("Makes correlations between Columns")
        st.code(code_snipet10, language='python')
        st.image("plotdiff.png", caption = "Example of Plot_diff use ", use_column_width=True)
    
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.subheader("Dealing with imbalanced datasets: SMOTE")
    st.text("An oversampling technique, to balance imbalanced datasets before training \nclassification models. It evaluates the accuracy, precision, recall, and F1-score\n of the models to compare results with and without balancing.")
    with st.expander("SMOTE:"):
        st.code(code_snipet8, language='python')
        st.subheader("Smote's Metrics Comparison")
        st.image("smote1.png", caption = "Metrics Comparison", use_column_width=True)
        st.subheader("Relevance of Caracteristics comparing to Smote")
        st.image("colunassmote.png", caption = "Graph", use_column_width=True)


with modelTraining:
    
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.header("**Machine Learning**")
    
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.subheader("* DT")

    st.text("\n")
    st.text("\n")
    st.image("dt.png", caption = "Decision Tree", use_column_width=True)
    st.markdown("* The classificator Decision Tree is trained and evaluated.")
    st.markdown("* Croos validation is realized to obtain medium accuracy")
    st.markdown("* ROC and AUC curves are calculated")
    st.markdown("* Confusion matrix and accuracy matrix are plotted")

    with st.expander("DT:"):
        st.code(code_snipet9, language='python')

    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.subheader("* KNN")
    
    st.text("\n")
    st.text("\n")

    st.image("knn.png", caption = "K-Nearest Neighbours", use_column_width=True)
    st.markdown("* K-nearest neighbors (KNN) is a simple algorithm for classification and regression.")
    st.markdown("* It predicts based on the closest data points.")
    st.markdown("* Performance depends on the number of neighbors and distance metric.")

    with st.expander ("KNN code:"):
        st.code(code_snipet11, language = 'python')

    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.subheader("Random Forest Classifier:")
    
    st.text("\n")
    st.text("\n")

    st.image("randomforest.png", caption = "Random Forest", use_column_width=True)
    st.markdown("* Random forest is an ensemble learning method for classification and regression.")
    st.markdown("* It builds multiple decision trees and merges their outputs for more accurate predictions.")
    st.markdown("* It reduces overfitting and improves performance compared to individual decision trees.")

    with st.expander ("Random Forest code:"):
        st.code(code_snipet12, language = 'python')

    
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.header("Data Evaluation")

    st.subheader("KNN, DT and Random Forest")
    st.text("Evaluation between KNN ,DT and Random Forest models")
    with st.expander("Data Evaluation Code"):
        st.code(code_snipet13, language='python')
    
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.markdown("* **ROC Curve comparison**")
    st.text("\n")
    st.text("\n")
    st.image("curvecomparisondataevaluation.png", caption = "Random Forest", use_column_width=True) 
    
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.markdown("* **Confusion Matrix**")
    st.text("\n")
    st.text("\n")
    st.image("confusionmatrix.png", caption = "Matrix", use_column_width=True)

    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.markdown("* **Cross Validation accuracies**")
    st.text("\n")
    st.text("\n")
    st.image("crossvalidation.png", caption = "Matrix", use_column_width=True)

    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.markdown("* **Performance during Learning**")
    st.text("\n")
    st.text("\n")
    st.image("learningcurve.png", caption = "Curves", use_column_width=True)

    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.markdown("* **Final comparison between algorithms**")
    st.text("\n")
    st.text("\n")
    st.image("comparacaodosmodelos.png", caption = "Final comparison", use_column_width=True)
    st.markdown("* **Holdout:**")
    st.text("Simple and quick, can result in unstable evaluations due to the only data division")
    st.text("\n")
    st.text("\n")
    st.markdown("* **K-Fold Cross-Validation:**")
    st.text("More robust and trustworthy, average evaluation of different k models and requires additional computer time.")
    st.text("\n")
    st.text("\n")
    st.markdown("* **Leave-One-Out Cross-Validation (LOOCV):**")
    st.text("Returns an extremelly precise evaluation,it´s extremelly CPU expensive and it´s useful for small amounts of data where the cost is manageable")

    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.text("\n")

    st.title("End of Project")
    st.text("Work produced by Daniela Leitão, Gonçalo Cruz and Leonor Rodrigues")