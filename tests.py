import json
import requests
import os
import pytest

def pytest_sessionstart(session):
    session.results = dict()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        item.session.results[item] = result

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 3 - 1 == 2
    
def pytest_sessionfinish(session, exitstatus):
    print()
    print('run status code:', exitstatus)
    passed_amount = sum(1 for result in session.results.values() if result.passed)
    failed_amount = sum(1 for result in session.results.values() if result.failed)
    print(f'there are {passed_amount} passed and {failed_amount} failed tests')

    webhook_url = os.getenv('WEBHOOK_URL')
    if webhook_url:
        result_dict = {'exit_status': exitstatus, 'passed_amount': passed_amount, 'failed_amount': failed_amount}
        test_results = json.dump(result_dict)
        response = requests.post(webhook_url, json=test_results)
    else:
        print("WEBHOOK_URL environment variable not set.")
