import pandas as pd
import requests
import sys
import os

sys.path.append(os.getcwd())
from logs.constant import Constant
from src.main import LogEntry


class LogSimulator:
    def __init__(self, log_file_path: str, api_url: str, batch_size: int):
        self.log_file_path = log_file_path
        self.api_url = api_url
        self.batch_size = batch_size

    
    def send_log(self, log_entry: LogEntry):
        """
            Send a log to the api
        """
        try:
            response = requests.post(
                url=self.api_url,
                json={
                    "timestamp": log_entry.timestamp,
                    "user_ip": log_entry.user_ip,
                    "method": log_entry.method,
                    "status_code": log_entry.status_code,
                    "end_point": log_entry.end_point,
                    "response_time": log_entry.response_time
                }
            )
            print(f"Log {response.json()} ajouté avec succès")
        except Exception as e:
            print("Echen de l'envoi du log: ", e)

    def simulate(self):
        """ Read from the log file and send each log to the API"""

        print("Début de la simulation................")
        log_df = pd.read_csv(self.log_file_path)
        log_df_batch = log_df.sample(self.batch_size)

        for idx, log_entry in log_df_batch.iterrows():
            self.send_log(log_entry=LogEntry(
                timestamp=log_entry.timestamp,
                user_ip=log_entry.user_ip,
                method=log_entry.method,
                status_code=str(log_entry.status_code),
                end_point=log_entry.end_point,
                response_time=log_entry.response_time
            ))

        print("Fin de la simulation..................")


if __name__ == '__main__':
    log_simulator = LogSimulator(
        log_file_path=Constant.LOGS_DATASET_FILE_NAME,
        api_url="http://127.0.0.1:8500/log",
        batch_size=20000
    )

    log_simulator.simulate()