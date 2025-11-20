import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#1. Importar dataset
df = pd.read_csv('banana_quality.csv')

#3. Estatísticas gerais da base
print("\nEstatísticas Descritivas")
print(df.describe().T)

# 4./5. Transformação colunas - qualidade -> 0/1
le = LabelEncoder()
df['Quality01'] = le.fit_transform(df['Quality'])
nomeclasse = le.classes_

print("\n\nQualidade: 0 = Bad, 1 = Good")

#4. Transformação linhas - retirar <-4 ou >4
nmrs = df.select_dtypes(include=np.number).columns.tolist()

mask = (df[nmrs] > -4).all(axis=1) & (df[nmrs] < 4).all(axis=1)

df_filtered = df[mask].copy()

lns_removidas = len(df) - len(df_filtered)
df = df_filtered

print("\n\nLinhas antes da remoção: ", len(df) + lns_removidas," \nLinhas após a remoção: ", len(df))
print("Linhas removidas: ", lns_removidas)

X = df.drop(['Quality', 'Quality01'], axis=1) 
y = df['Quality01']                       

#6. Subconjuntos
Xtreinovalid, Xteste, Ytreinovalid, Yteste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtreino, Xvalid, Ytreino, Yvalid = train_test_split(
    Xtreinovalid, Ytreinovalid, test_size=0.25, random_state=42, stratify=Ytreinovalid
)

print("\n\nTotal: ", len(X) ,"amostras")
print("Treinamento: ", len(Xtreino) ,"amostras")
print("Validação: ", len(Xvalid) ,"amostras")
print("Teste: ", len(Xteste) ,"amostras")

scaler = StandardScaler()
treino = scaler.fit_transform(Xtreino)
valid = scaler.transform(Xvalid)
teste = scaler.transform(Xteste)

plt.figure(figsize=(6, 4))
sns.countplot(x='Quality', data=df, color='yellow')
plt.title('Qualidade das Bananas')
plt.show()

#7. Treinamento e avaliação do modelo
rf_modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_modelo.fit(treino, Ytreino)

Yvalid_pred = rf_modelo.predict(valid)

#8. Acurácia e matriz de confusão
mat_conf = confusion_matrix(Yvalid, Yvalid_pred)
acuracia = accuracy_score(Yvalid, Yvalid_pred)

print("\n\nAcurácia: ", acuracia ,"%")

plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf, annot=True, fmt='d', cmap='inferno', 
            xticklabels=nomeclasse, yticklabels=nomeclasse)
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Predição')
plt.show()

#9. Predição  
exemplo_banana = Xteste.iloc[[0]] 
y_real = Yteste.iloc[0]

exemplo_banana = scaler.transform(exemplo_banana)
pred_dec = rf_modelo.predict(exemplo_banana)[0]

pred = nomeclasse[pred_dec]
real = nomeclasse[y_real]

print("\n\n", pd.DataFrame(exemplo_banana, columns=Xteste.columns, index=['Amostra']).to_string())

print("\n\nQualidade Predita: ", pred)
print("Qualidade Real: ", real)