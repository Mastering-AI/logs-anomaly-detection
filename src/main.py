from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any
from celery import Celery
import redis
import os  
import json
from pathlib import Path
import pandas as pd
from logs.constant import Constant
from logs.preprocessing import LogPreprocessor 
from logs.training import AnomalyDetector
import time


THRESHOLD = 1000
REDIS_LOG_KEY = "log_buffer"
REDIS_COUNTER_KEY = "log_counter"
REDIS_PREFIX_PREDICTION_KEY = "prediction"

app = FastAPI(
    title="Logs anomaly detection API",
    description="Api for detecting anomalies in log data",
    version="1.0.0"
)

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0
)

celery_app = Celery(
    "anomaly_detection",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

class LogEntry(BaseModel):
    timestamp: str
    user_ip: str 
    method: str 
    status_code: str 
    end_point: str
    response_time: int
        

@celery_app.task(name="process_batch_prediction")
def process_batch_prediction(log_data_json: str):
    
    try:
    
        log_entries = json.loads(log_data_json)
        
        df = pd.DataFrame(log_entries)
        
        preprocessor = LogPreprocessor()
        
        x_preprocessed, x = preprocessor.fit_transform(df)
        
        detector = AnomalyDetector()
        
        predictions = detector.predict(x_preprocessed)
        
        scores = detector.compute_anomaly_scores(x_preprocessed)
        
        isolation_forest_preds = predictions["isolation_forest"].tolist()
        one_class_svm_preds = predictions["one_svm"].tolist()
        
        anomalies_count = sum(1 for pred in isolation_forest_preds if pred == -1)
        
        result = {
            "isolation_forest_preds" : isolation_forest_preds,
            "one_class_svm_preds": one_class_svm_preds,
            "scores" : {
                    "isolation_forest_scores" : scores["isolation_forest"].tolist,
                    "one_class_svm_scores" : scores["one_svm"].tolist
                    
                },
            "anomalies_count" : anomalies_count,
            "timestamp": time.time()
            
        }
        
        prediction_id = int(time.time())
        
        redis_client.setex(
            f"{REDIS_PREFIX_PREDICTION_KEY}{prediction_id}",
            86400,
            json.dumps(result)
        )
        
        redis_client.lpush("prediction_list", prediction_id)
        
        return {
            "message": f" {prediction_id} prediction completed",
            "status": "success"
        }
    except Exception as e:
        return {
            "message": f"{str(e)}",
            "status": "error"
        }
    

@app.get("/")
def root():
    
    return {
        "message": "API is ready !"
    }
    
    
@app.get("/status")
def check_model():

    isolation_forest_exists = os.path.exists(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
    one_class_svm_exists = os.path.exists(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)
    
    if not (isolation_forest_exists and one_class_svm_exists):
        raise HTTPException(
            status_code=503,
            detail="Models are not available"
        )
    else:
        
        buffer_size = redis_client.llen(REDIS_LOG_KEY) or 0
    
        # Get predictions count
        predictions_count = redis_client.llen("prediction_list") or 0
        
        return {
            "status": "operational",
            "models_loaded": isolation_forest_exists and one_class_svm_exists,
            "buffer_size": buffer_size,
            "threshold": THRESHOLD,
            "predictions_available": predictions_count
        }

        
        
@app.post("/log")
def add_log(log_entry: LogEntry):
    """add a single log entry to the redis Buffer
    """
    
    log_json = json.dumps(log_entry.dict())
    
    log_id = redis_client.incr(REDIS_COUNTER_KEY)
    
    redis_client.lpush(REDIS_LOG_KEY, log_json)
    
    buffer_size = redis_client.llen(REDIS_LOG_KEY)
    
    if buffer_size > THRESHOLD:
        
        logs_json_list = redis_client.lrange(REDIS_LOG_KEY, 0, -1)
        
        logs_data = [json.loads(log_json.decode('utf-8')) for log_json in logs_json_list]
        
        redis_client.delete(REDIS_LOG_KEY)

        task = process_batch_prediction.delay(json.dumps(logs_data))
        
        return {
            "message": "Log added and batch processing triggered",
            "task": task.id,
            "log_id": log_id
        }
    
    return {
        "message": "Log added to buffer",
        "buffer_size": buffer_size,
        "log_id": log_id,
        "threshold": THRESHOLD
    }
    
@app.get("/predictions", response_model=Dict[str, Any])
def get_predictions():
    """
    Get IDs of all available predictions
    """
    # Get list of prediction IDs
    prediction_ids = redis_client.lrange("prediction_list", 0, -1)
    prediction_ids = [int(pid.decode('utf-8')) for pid in prediction_ids]
    
    return {
        "prediction_count": len(prediction_ids),
        "prediction_ids": prediction_ids
    }
    

@app.get("/predictions/{prediction_id}", response_model=Dict[str, Any])
def get_prediction_by_id(prediction_id: int):
    """
    Get a specific prediction result by ID
    """
    # Get prediction from Redis
    prediction_json = redis_client.get(f"{REDIS_PREFIX_PREDICTION_KEY}{prediction_id}")
    
    if not prediction_json:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Parse prediction data
    prediction_data = json.loads(prediction_json)
    
    return prediction_data

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    
    """
    Get status of a Celery task
    """
    task = celery_app.AsyncResult(task_id)
    
    response = {
        "task_id": task_id,
        "status": task.status
    }
    
    if task.ready():
        response["result"] = task.result
    
    return response

@app.get("/clear_db")
def clear_redis_cache():
    
    redis_client.flushdb()
    
    return {
        "message": "db cleared"   
    }
    

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)