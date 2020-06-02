import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def kde_evaluate(x, x_grid):

    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.05, 0.1, 100)},
                        cv=20)  # 20-fold cross-validation
    grid.fit(x)
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid))

    return pdf
