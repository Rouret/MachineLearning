import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Créer les données d'apprentissage
rng = np.random.RandomState(1)
X = np.sort(5*rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()

regr_1 = DecisionTreeRegressor(max_depth=100)
regr_1.fit(X, y)

# Prédiction
# start/stop/setp
X_test = np.arange(0.0, 5.0, 0.001).reshape(-1,1)
y_1 = regr_1.predict(X_test)

# Affichage des résultats
plt.figure()
plt.scatter(X, y, c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()