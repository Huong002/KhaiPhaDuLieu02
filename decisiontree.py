import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # Thêm thư viện joblib để lưu mô hình

def importdata():
    balance_data = pd.read_csv("diabetes_data_upload.csv", sep=',', header=0)

    label_encoder = LabelEncoder()
    for column in balance_data.columns:
        if balance_data[column].dtype == 'object':
            balance_data[column] = label_encoder.fit_transform(balance_data[column])

    print("Dataset after encoding: ")
    print(balance_data.head())
    return balance_data

def splitdataset(balance_data):
    X = balance_data.iloc[:, :-1]  
    Y = balance_data.iloc[:, -1]   

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42 
    )
    return X, Y, X_train, X_test, y_train, y_test

def train_using_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [5, 10, 20, None], 
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 4]  
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Lưu mô hình tốt nhất vào tệp .pkl
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'random_forest_model.pkl')
    print("Model saved as 'random_forest_model.pkl'")

    return best_model

def load_model(file_path):
    return joblib.load(file_path)

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
    print("Report:\n", classification_report(y_test, y_pred))

# Driver code
def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # Huấn luyện và lưu mô hình
    clf_rf = train_using_random_forest(X_train, y_train)

    # Sử dụng mô hình đã lưu
    loaded_model = load_model('random_forest_model.pkl')

    print("\nResults Using Random Forest (Loaded Model):")
    y_pred_rf = prediction(X_test, loaded_model)
    cal_accuracy(y_test, y_pred_rf)

if __name__ == "__main__":
    main()
