from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import os
from io import StringIO

app = Flask(__name__)
CORS(app)  # Cho phép tất cả các nguồn

# Đường dẫn đầy đủ đến các tệp đã lưu
model_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'stacking_model.pkl')
train_data_path = os.path.join('C:', 'Users', 'LENOVO', 'Downloads', 'train_data.csv')

# Tải mô hình và các đối tượng tiền xử lý đã lưu
model = joblib.load(model_path)
train_data_1 = pd.read_csv(train_data_path)

# Hàm chuyển đổi boolean
def convert_boolean(data):
    data['IsHoliday'] = data['IsHoliday'].fillna(False)  # Thay thế NaN bằng False
    data['IsHoliday'] = data['IsHoliday'].map({False: 0, True: 1}).astype('int')
    return data

# Hàm thay thế giá trị NAN
def replace_nan(data):
    columns_to_fill = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    data[columns_to_fill] = data[columns_to_fill].fillna(0)
    return data

# Hàm trích xuất thông tin ngày tháng
def extract_date_info(df):
    df['Date2'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date2'].dt.day.astype('int')
    df['Month'] = df['Date2'].dt.month.astype('int')
    df['Year'] = df['Date2'].dt.year.astype('int')
    df['WeekOfYear'] = df['Date2'].dt.isocalendar().week.astype('int')
    df['Quarter'] = df['Date2'].dt.quarter.astype('int')
    df = df.drop(columns=['Date2'])
    return df

# Hàm xử lý thông tin markdown
def markdown_info(df):
    df['MarkDown'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
    df.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1, inplace=True)
    return df

# Hàm xử lý thông tin ngày lễ
def isholiday(df):
    holiday_weeks = [1, 6, 36, 47, 52]
    df.loc[df['WeekOfYear'].isin(holiday_weeks), 'IsHoliday'] = 1
    return df

def num_cat_cols(data):
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes('object').columns.tolist()
    return numeric_cols, categorical_cols

# API để dự đoán doanh số bán hàng từ tệp CSV
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        data = []

        # Đọc tệp CSV
        csv_file = StringIO(file.stream.read().decode('utf-8'))
        df = pd.read_csv(csv_file)

        train_data = convert_boolean(df)
        train_data = replace_nan(df)
        train_data = extract_date_info(train_data)
        train_data = markdown_info(train_data)
        train_data = isholiday(train_data)
        train_data = train_data.sort_values(by=['Date']).reset_index(drop=True)
        numeric_cols = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Day', 'Month', 'Year', 'WeekOfYear', 'Quarter', 'MarkDown']
        train_size = int(.75 * len(train_data))
        train_df, val_df = train_data[:train_size], train_data[train_size:]

        input_cols = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size', 'Day', 'Month', 'Year', 'WeekOfYear', 'Quarter', 'MarkDown']
        target_col = 'Weekly_Sales'
        train_inputs = train_df[input_cols].copy()
        train_targets = train_df[target_col].copy()
        val_inputs = val_df[input_cols].copy()
        val_targets = val_df[target_col].copy()

        numeric_cols, categorical_cols = num_cat_cols(train_inputs)

        imputer = SimpleImputer(strategy='mean').fit(train_inputs[numeric_cols])
        train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])

        scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
        encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
        val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

        X_train = train_inputs[numeric_cols + encoded_cols]
        X_val = val_inputs[numeric_cols + encoded_cols]

        model.fit(X_train, train_targets.values.ravel())

        prediction = model.predict(X_train)
        val_preds = model.predict(X_val)

        train_mae = mean_absolute_error(train_targets, prediction)
        train_rmse = mean_squared_error(train_targets, prediction, squared=False)
        train_r2 = r2_score(train_targets, prediction)

        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_r2 = r2_score(val_targets, val_preds)

        scores = {
            'Metric': ['MAE', 'RMSE', 'R2'],
            'Training': [round(train_mae, 2), round(train_rmse, 2), round(train_r2, 2)],
            'Validation': [round(val_mae, 2), round(val_rmse, 2), round(val_r2, 2)]
        }

        result = []
        for i, pred in enumerate(prediction):
            result.append({
                'Store': int(df.iloc[i]['Store']),
                'Prediction': float(pred)
            })

        return jsonify({'predictions': result, 'scores': scores})

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