import pytest
import joblib
import pandas as pd


def test_model_load():
    model = joblib.load('model/model.pkl')
    assert model is not None


def test_model_prediction():
    model = joblib.load('model/model.pkl')
    data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = model.predict(data)
    assert prediction[0] in ['setosa', 'versicolor', 'virginica']


if __name__ == '__main__':
    pytest.main()
