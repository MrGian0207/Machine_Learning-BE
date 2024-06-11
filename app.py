from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import csv
import os
from io import StringIO
import datetime

app = Flask(__name__)
CORS(app)  # Cho phép tất cả các nguồn

# Đường dẫn đầy đủ đến các tệp đã lưu
model_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'stacking_model.pkl')
imputer_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'imputer.pkl')
scaler_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'scaler.pkl')
encoder_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'encoder.pkl')
train_target_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'train_targets.csv')
X_train_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'X_train.csv')

# Tải mô hình và các đối tượng tiền xử lý đã lưu
model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

# Xác định các cột số và cột phân loại
numeric_cols = ['Store', 'Dept','IsHoliday','Temperature','Fuel_Price','CPI', 'Unemployment','Size','Day','Month','Year','WeekOfYear','Quarter','MarkDown']
categorical_cols = ['Type']  # Thay thế bằng các cột phân loại của bạn nếu có

# Hàm tiền xử lý dữ liệu
def preprocess_data(data):
    # Chuyển đổi dữ liệu thành DataFrame
    df = pd.DataFrame(data)
    
    # Imputation: Thay thế giá trị thiếu bằng giá trị trung bình cho các cột số
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    
    # Scaling: Chuẩn hóa dữ liệu cho các cột số
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    # Encoding: Mã hóa các biến phân loại
    if categorical_cols:
        encoded_data = encoder.transform(df[categorical_cols])
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        df[encoded_cols] = encoded_data
    
    # Đảm bảo rằng DataFrame có cùng tên cột với dữ liệu huấn luyện
    df = df[numeric_cols + encoded_cols]
    print(df)
    return df

# Hàm chuyển đổi boolean
def convert_boolean(data):
    for row in data:
        row['IsHoliday'] = 1 if row['IsHoliday'] else 0
    return data

# Hàm thay thế giá trị NAN
def replace_nan(data):
    columns_to_fill = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for row in data:
        for col in columns_to_fill:
            if row[col] is None or row[col] == '':
                row[col] = 0
    return data

# Hàm trích xuất thông tin ngày tháng
def extract_date_info(data):
    for row in data:
        date = datetime.datetime.strptime(row['Date'], '%Y-%m-%d')
        row['Day'] = date.day
        row['Month'] = date.month
        row['Year'] = date.year
        row['WeekOfYear'] = date.isocalendar()[1]
        row['Quarter'] = (date.month - 1) // 3 + 1
    return data

# Hàm xử lý thông tin markdown
def markdown_info(data):
    for row in data:
        row['MarkDown'] = sum(float(row.get(f'MarkDown{i}', 0)) for i in range(1, 6))
        for i in range(1, 6):
            row.pop(f'MarkDown{i}', None)
    return data

# Hàm xử lý thông tin ngày lễ
def is_holiday(data):
    holiday_weeks = [1, 6, 36, 47, 52]
    for row in data:
        if row['WeekOfYear'] in holiday_weeks:
            row['IsHoliday'] = 1
    return data

# API để dự đoán doanh số bán hàng từ tệp CSV
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = []

        # Đọc tệp CSV
        csv_file = StringIO(file.stream.read().decode('utf-8'))
        reader = csv.DictReader(csv_file)

        train_targets = pd.read_csv(train_target_path)
        # X_train = pd.read_csv(X_train_path)
        for row in reader:
            data.append(row)


        df = pd.DataFrame(data)

        df = df.astype(float)

        model.fit(df, train_targets.values.ravel())
        predictions = model.predict(df)

        result = []
        for i, prediction in enumerate(predictions):
            result.append({
                'Store': data[i]['Store'],
                'Prediction': prediction
            })

        return jsonify(result)
       
    except Exception as e:
        return jsonify({'error': str(e)})

# API để tải xuống mô hình và các đối tượng tiền xử lý
@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    file_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)