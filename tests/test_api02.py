import pytest
from flask import Flask
from pro7.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_get_value_route(client):
    response = client.get('/get_value/100002')
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    assert b'<html>' in response.data
    assert b'Client ID: 100002' in response.data
    assert b'ACTIVE_DAYS_CREDIT_ENDDATE_MIN' in response.data
    assert b'AMT_ANNUITY' in response.data
    assert b'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN' in response.data
    assert b'BURO_CREDIT_ACTIVE_Closed_MEAN' in response.data
    assert b'CODE_GENDER' in response.data
    assert b'DAYS_BIRTH' in response.data
    assert b'DAYS_EMPLOYED' in response.data
    assert b'EXT_SOURCE_2' in response.data
    assert b'EXT_SOURCE_3' in response.data
    assert b'FLAG_OWN_CAR' in response.data
    assert b'NAME_EDUCATION_TYPE_Highereducation' in response.data
    assert b'ORGANIZATION_TYPE_Selfemployed' in response.data
    assert b'PAYMENT_RATE' in response.data
    assert b'PREV_CODE_REJECT_REASON_XAP_MEAN' in response.data
    assert b'PREV_NAME_CONTRACT_STATUS_Refused_MEAN' in response.data


