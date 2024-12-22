from flask import Flask, request, render_template
import numpy as np
import joblib


app = Flask(__name__,template_folder="template")

# Từ điển chứa các tệp mô hình
models = {
    'Decisiontree': 'random_forest_model.pkl',
    'Logistic_regression': 'logistic_regression.pkl',
}

@app.route('/')
def home():
    return render_template('index.html',form_data={}, prediction_text=None, class_text=None)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Lấy thông tin model từ form
    model_choice = request.form.get('model')
    model_path = models.get(model_choice)

    if not model_path:
        return render_template('index.html', prediction_text='Vui lòng chọn mô hình hợp lệ.', class_text='error', active_section='predict',form_data=request.form)

    # Load mô hình
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi khi tải mô hình: {str(e)}', class_text='error', active_section='predict',form_data=request.form)

    # Chuẩn bị dữ liệu đầu vào
    try:
        input_data = np.array([
            int(request.form.get('Age', 0)),  # Tuổi
            1 if request.form.get('Gender', 'no') == 'male' else 0,  # Giới tính: Male = 1, Female = 0
            1 if request.form.get('Polyuria', 'no') == 'yes' else 0,
            1 if request.form.get('Polydipsia', 'no') == 'yes' else 0,
            1 if request.form.get('chsudden_weight_loss', 'no') == 'yes' else 0,
            1 if request.form.get('weakness', 'no') == 'yes' else 0,
            1 if request.form.get('Polyphagia', 'no') == 'yes' else 0,
            1 if request.form.get('Genital_thrush', 'no') == 'yes' else 0,
            1 if request.form.get('visual_blurring', 'no') == 'yes' else 0,
            1 if request.form.get('Itching', 'no') == 'yes' else 0,
            1 if request.form.get('Irritability', 'no') == 'yes' else 0,
            1 if request.form.get('delayed_healing', 'no') == 'yes' else 0,
            1 if request.form.get('partial_paresis', 'no') == 'yes' else 0,
            1 if request.form.get('muscle_stiffness', 'no') == 'yes' else 0,
            1 if request.form.get('Alopecia', 'no') == 'yes' else 0,
            1 if request.form.get('Obesity', 'no') == 'yes' else 0,
        ]).reshape(1, -1)
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Lỗi dữ liệu đầu vào: {str(e)}', class_text='error', active_section='predict',form_data=request.form)

    # Dự đoán
    try:
        predicted_target = model.predict(input_data)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi dự đoán: {str(e)}', class_text='error', active_section='predict',form_data=request.form)

    # Xác định kết quả
    if predicted_target[0] == 1:
        result = 'Bạn có nguy cơ mắc bệnh tiểu đường. Hãy tham khảo ý kiến bác sĩ!'
        class_css = 'disease'
    else:
        result = 'Sức khỏe của bạn ổn định. Hãy kiểm tra sức khỏe định kỳ!'
        class_css = 'no-disease'

    return render_template('index.html', prediction_text=result, class_text=class_css, active_section='predict',  form_data=request.form)

if (__name__ == '__main__'):
    app.run(debug=True)
