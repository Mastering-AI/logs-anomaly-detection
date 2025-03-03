import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

sys.path.append(os.getcwd())
from logs.preprocessing import LogPreprocessor
from logs.constant import Constant


class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            n_jobs=-1, verbose=1, contamination=.05
        )
        self.one_svm = OneClassSVM(verbose=1, nu=.05)

    def train_models(self, X:np.ndarray):
        """
            Train isolation forest and one class SVM models
        """
        self.isolation_forest.fit(X)
        self.one_svm.fit(X)

        joblib.dump(self.isolation_forest, Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        joblib.dump(self.one_svm, Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

    def predict(self, X:np.ndarray) -> dict[str, np.ndarray]:
        """
            Make predictions on X using isolation forest and one class SVM models.
        """
        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        return {
            "isolation_forest": isolation_forest.predict(X),
            "one_svm": one_svm.predict(X)
        }

    def compute_anomaly_scores(self, X:np.ndarray) -> dict[str, np.ndarray]:
        """
            Compute anomaly score per model (Isolation forest and One class SVM)
        """
        isolation_forest: IsolationForest = joblib.load(Constant.ISOLATION_FOREST_MODEL_FILE_NAME)
        one_svm: OneClassSVM = joblib.load(Constant.ONE_CLASS_SVM_MODEL_FILE_NAME)

        return {
            "isolation_forest": isolation_forest.decision_function(X),
            "one_svm": one_svm.decision_function(X)
        }

    def evaluate_models(self, X: np.ndarray):
        """
            Evaluate Isolation forest and One class SVM models
        """

        scores = self.compute_anomaly_scores(X)

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.hist(scores["isolation_forest"], bins=20, color='blue', alpha=.7)
        ax1.set_title("Histogramme Isolation forest")

        ax2.hist(scores["one_svm"], bins=20, color='green', alpha=.7)
        ax2.set_title("Histogramme One Class SVM")

        plt.xlabel("Scores d'anomalies")
        plt.ylabel("Fréquences")
        plt.tight_layout()
        plt.show()

        predictions = self.predict(X)

        outlier_isolation_forest_ratio = np.mean(predictions["isolation_forest"]==-1)
        outlier_svm_ratio = np.mean(predictions["one_svm"]==-1)

        print(f"Ratio d'outlier Isolation Forest: {outlier_isolation_forest_ratio:.2%}")
        print(f"Ratio d'outlier One Class SVM: {outlier_svm_ratio:.2%}")

    

if __name__ == "__main__":
    df = pd.read_csv(Constant.LOGS_DATASET_FILE_NAME)

    preprocessor = LogPreprocessor()

    df_train, df_test = preprocessor.split_dataset(df)

    print("Prétraitement des données...")
    X_train, df_train_engineered = preprocessor.fit_transform(df_train)
    X_test, df_test_engineered = preprocessor.fit_transform(df_test)

    detector = AnomalyDetector()

    print("Entraînement des modèles")
    detector.train_models(X_train)

    print("Evaluation des modèles sur le Test set")
    detector.evaluate_models(X_test)

    predictions = detector.predict(X_test)