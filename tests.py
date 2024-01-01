import json
import requests
import os
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 3 - 1 == 2


webhook_url = os.getenv('WEBHOOK_URL')

if webhook_url:
    test_results = {"status": "success", "message": "Tests passed successfully"}

    #response = requests.post(webhook_url, json=test_results)
else:
    print("WEBHOOK_URL environment variable not set.")
