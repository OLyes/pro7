import pytest
from flask import Flask
from pro7.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_predict_route(client):
    response = client.get('/predict/100002')
    assert response.status_code == 200
    data = response.json
    assert 'client_id' in data
    assert 'probability' in data
    assert 'threshold_value' in data
    assert 'loan_status' in data
    assert isinstance(data['client_id'], int)
    assert isinstance(data['probability'], float)
    assert isinstance(data['threshold_value'], float)
    assert isinstance(data['loan_status'], str)

