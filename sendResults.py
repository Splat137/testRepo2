import os
import json
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resultlog', dest='logPath', default=None)
args = parser.parse_args()
logPath = args.logPath

if logPath:
    if os.path.exists(logPath):
        with open(logPath, 'r') as logFile:    
            lines = logFile.readlines()
            jsonList = [json.loads(line.strip()) for line in lines]
            jsonObject = {"data": jsonList}
            
            webhookUrl = os.getenv('WEBHOOK_URL')
            if webhookUrl:
                requests.post(webhookUrl, json=jsonObject)
            else:
                print("WEBHOOK_URL environment variable not set.")
    else:
        print(f"Log file not found at: {log_path}")
else:
    print("--resultlog option not specified.")
