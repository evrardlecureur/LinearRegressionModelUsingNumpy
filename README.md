# LinearRegressionModelUsingNumpy
Ce projet est le fruit d'une démarche d'apprentissage : re-coder entièrement un modèle de Régression Linéaire par la méthode de la Descente de Gradient (LR.py).

L'objectif principal était de comprendre en profondeur les mécanismes et les calculs matriciels du Machine Learning en utilisant uniquement NumPy pour la logique de base. Le code est ensuite validé par un benchmark contre la référence professionnelle : la classe LinearRegression de Scikit-learn qui n'utilise qu'une étape de part sa nature analytique.

Nous comparons les performances (RMSE) après avoir entraîné les deux modèles sur un sous-ensemble standardisé du jeu de données Housing Prices de kaggle.


| Modèle | RMSE | Temps d'Exécution | Note |
| :--- | :--- | :--- | :--- |
| **scikit-learn** | 39763.30 | 0.020s | Solution analytique (1 calcul) |
| **Mon Modèle Custom** | 39763.17 | 0.038s | Solution obtenue après 1000 itérations |


Pour lancer le projet 

git clone [Lien de votre dépôt GitHub]

cd [Nom du dépôt]

pip install -r requirements.txt

python test.py
