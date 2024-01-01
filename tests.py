import json
import requests
import os

webhook_url = os.getenv('WEBHOOK_URL')

if webhook_url:
    # Replace this with your test results or any data you want to send
    test_results = {"status": "success", "message": "Tests passed successfully"}

    response = requests.post(webhook_url, json=test_results)

    if response.status_code == 200:
        print("Test results sent successfully.")
    else:
        print(f"Failed to send test results. Status code: {response.status_code}, Response: {response.text}")
else:
    print("WEBHOOK_URL environment variable not set.")
