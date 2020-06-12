"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class linear_regr:

    def __init__(self, criterion = "mse"):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.criterion = criterion.lower()
        self.loss_function = mean_absolute_error if self.criterion == "mae" else mean_squared_error

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return self.loss_function(y, y_pred)

