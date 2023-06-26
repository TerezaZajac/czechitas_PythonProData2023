"""
Uvažuj, že chceš přijímat lidi do organizace, která vyžaduje vysokou fyzickou výkonnost. Tvou snahou je zkrátit a zefektivnit přijímací proces. Zkus tedy zjistit, nakolik přesné je zařazení jedinců do výkonnostních tříd bez nutnosti měření jejich výkoknu při vykonání jednotlivých cviků. Využij tedy všechny vstupní proměnné s výjimkou sit and bend forward_cm, sit-ups counts a broad jump_cm.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data_dp = pd.read_csv('czechitas_PythonProData2023/bodyPerformance.csv')
#print(data_dp.head())


y = data_dp['class']
categorical_columns = ['gender']
numeric_columns = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic', 'gripForce']
numeric_data = data_dp[numeric_columns].to_numpy()

encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data_dp[categorical_columns])
encoded_columns = encoded_columns.toarray()

X = np.concatenate([encoded_columns, numeric_data], axis=1)
#print(X)

"""
K rozřazení jedinců do skupin využij rozhodovací strom a jeden ze zbývajících dvou algoritmů probíraných na lekcích (tj. K Nearest Neighbours nebo Support Vector Machine). Rozhodovacímu stromu omez maximální počet pater na 5 a poté si zobraz graficky a vlož ho do Jupyter notebooku nebo jako obrázek ve formátu PNG jako součást řešení.
"""

#rozdeleni dat na trenovaci a testovaci
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#rozhodovaci strom
clf_dtc = DecisionTreeClassifier(max_depth=5)
clf_dtc = clf_dtc.fit(X_train, y_train)
y_pred = clf_dtc.predict(X_test)

dot_data = StringIO()
export_graphviz(clf_dtc, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('tree_dp_1.png')


"""
Vytvoř matici záměn pro rozhodovací strom. Kolik jedinců s nejvyšší fyzickou výkonností (tj. ze skupiny A) bylo klasifikování správně? Kolik pak bylo zařazeno do skupin B, C a D? Uveď výsledky do komentáře v programu nebo do buňky v Jupyter notebooku.
"""
ConfusionMatrixDisplay.from_estimator(clf_dtc, X_test, y_test)
plt.title('Matice zamen')
#plt.show()


"""
Urči metriku accuracy pro rozhodovací strom a pro jeden ze dvou vybraných algoritmů. Který algoritmus si vedl lépe? Odpověď napiš do komentáře.
"""
print(accuracy_score(y_test, y_pred))
# hodnota metriky accuracy pro rozhodovaci strom je 44.89 % (0.4489795918367347)

#KNN
scaler = StandardScaler()
numeric_data = scaler.fit_transform(data_dp[numeric_columns])
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data_dp[categorical_columns])
encoded_columns = encoded_columns.toarray()
X = np.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred = clf_knn.predict(X_test)
ConfusionMatrixDisplay.from_estimator(clf_knn, X_test, y_test)
#plt.show()
print(accuracy_score(y_test, y_pred))
# hodnota metriky accurancy je 41.04 % (0.41040318566450973)
#KNN si vedl lepe



"""
Nyní uvažuj, že se rozhodneš testovat jedince pomocí jednoho ze cviků. Vyber cvik, který dle tebe nejvíce vypovídá o fyzické výkonnosti jedince. Porovnej, o kolik se zvýšila hodnota metriky accuracy pro oba algoritmy.
"""
#pridavam sloupec 'sit-ups counts''
numeric_columns2 = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic', 'gripForce', 'sit-ups counts']
numeric_data = data_dp[numeric_columns2].to_numpy()
X = np.concatenate([encoded_columns, numeric_data], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf_2 = DecisionTreeClassifier(max_depth=5)
clf_2.fit(X_train, y_train)
y_pred = clf_2.predict(X_test)
dot_data = StringIO()

export_graphviz(clf_2, out_file=dot_data, filled=True, feature_names=numeric_columns2 + list(encoder.get_feature_names_out()), class_names=["A", "B", "C", "D"])


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree_dp_2.png')
ConfusionMatrixDisplay.from_estimator(clf_2, X_test, y_test)
#plt.show()


print(accuracy_score(y_test, y_pred))
# hodnota metriky accurancy je 50.27 %


scaler = StandardScaler()
numeric_data = scaler.fit_transform(data_dp[numeric_columns2])
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data_dp[categorical_columns])
encoded_columns = encoded_columns.toarray()
X = np.concatenate([encoded_columns, numeric_data], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_3 = KNeighborsClassifier()
clf_3.fit(X_train, y_train)
y_pred = clf_3.predict(X_test)

print(accuracy_score(y_test, y_pred))
# hodnota metriky accurancy je 51.64 %



