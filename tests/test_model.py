import pytest
import joblib
import pandas as pd

def test_model_load():
    model = joblib.load('model.pkl')
    assert model is not None

def test_model_prediction():
    model = joblib.load('model.pkl')
    data = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=[
            'SepalLengthCm',
            'SepalWidthCm',
            'PetalLengthCm',
            'PetalWidthCm'
        ]
    )
    prediction = model.predict(data)
    assert prediction[0] in ['setosa', 'versicolor', 'virginica']

if __name__ == '__main__':
    pytest.main()
