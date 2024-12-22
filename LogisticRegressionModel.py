import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib  # Thêm thư viện joblib để lưu mô hình

# Import dữ liệu
def importdata():
    balance_data = pd.read_csv("diabetes_data_upload.csv", sep=',', header=0)

    # Encode các cột kiểu object thành số
    label_encoder = LabelEncoder()
    for column in balance_data.columns:
        if balance_data[column].dtype == 'object':
            balance_data[column] = label_encoder.fit_transform(balance_data[column])

    print("Dataset after encoding: ")
    print(balance_data.head())
    return balance_data

# Chia dữ liệu thành tập huấn luyện và kiểm tra
def splitdataset(balance_data):
    X = balance_data.iloc[:, :-1]  # Tất cả cột trừ cột cuối (đặc trưng)
    Y = balance_data.iloc[:, -1]   # Cột cuối cùng (nhãn)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42  # 20% dữ liệu để kiểm tra
    )
    return X, Y, X_train, X_test, y_train, y_test

# Huấn luyện mô hình Logistic Regression
def train_using_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],         # Hệ số điều chuẩn
        'solver': ['lbfgs', 'liblinear'],     # Các thuật toán tối ưu
        'max_iter': [100, 200, 300]           # Số lần lặp tối đa
    }

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=5, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)

    # Lưu mô hình tốt nhất vào tệp .pkl
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'logistic_regression.pkl')
    print("Model saved as 'logistic_regression_model.pkl'")

    return best_model

# Tải mô hình từ tệp
def load_model(file_path):
    return joblib.load(file_path)

# Dự đoán
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    return y_pred

# Tính độ chính xác
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
    print("Report:\n", classification_report(y_test, y_pred))

# Driver code
def main():
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

    # Huấn luyện và lưu mô hình Logistic Regression
    clf_lr = train_using_logistic_regression(X_train, y_train)

    # Sử dụng mô hình đã lưu
    loaded_model = load_model('logistic_regression.pkl')

    print("\nResults Using Logistic Regression (Loaded Model):")
    y_pred_lr = prediction(X_test, loaded_model)
    cal_accuracy(y_test, y_pred_lr)

if __name__ == "__main__":
    main()
