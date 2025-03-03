from datetime import datetime

class Constant:
    LOGS_START_DATE: datetime = datetime(year=2024, month=1, day=1, hour=0, minute=0, second=0)
    LOGS_END_DATE: datetime = datetime(year=2025, month=1, day=1, hour=0, minute=0, second=0)
    NUMBER_LOGS: int = 100000
    HTTP_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE"]
    API_ENDPOINTS: list[str] = ["/users", "/admin", "/login", "/data", "/metrics"]
    HTTP_NORMAL_CODES: list[str] = ["200", "201", "204"]
    HTTP_ERROR_CODES: dict[str, list[str]] = {
        "server_errors": ["500", "502", "503"],
        "client_errors": ["400", "401", "403", "404"],
        "timeout_errors": ["408", "504"]
    }

    NUMBER_OF_ANOMALY_INTERVALS: int = 5
    MIN_NUMBER_OF_ANOMALY_PER_INTERVAL: int = 500
    MAX_NUMBER_OF_ANOMALY_PER_INTERVAL: int = 1000
    NUMBER_OF_ANOMALY_IPS: int = 15
    LOGS_DATASET_FILE_NAME: str = "data/logs_dataset.csv"

    TRAIN_SET_RATIO: float = 0.85
    ISOLATION_FOREST_MODEL_FILE_NAME = "models/isolation_forest.joblib"
    ONE_CLASS_SVM_MODEL_FILE_NAME = "models/one_class_svm.joblib"

