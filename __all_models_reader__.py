import os
import pandas as pd
import math
from joblib import load
from openpyxl import Workbook
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#------------------
# Veri setini yükle
#------------------
data_2022 = pd.read_csv("2022_input_space.csv")
data_2021 = pd.read_csv("2021_input_space.csv")
data_2020 = pd.read_csv("2020_input_space.csv")
data_2017 = pd.read_csv("2017_input_space.csv")
data_2016 = pd.read_csv("2016_input_space.csv")
data_2015 = pd.read_csv("2015_input_space.csv")
data = pd.concat([data_2022, data_2021, data_2020, data_2017, data_2016, data_2015], ignore_index=True)

data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['Hour'] = data['Timestamp'].dt.hour
data.set_index('Timestamp', inplace=True)
y_ = data["llj_level"]
X_ = data[["u10_value", "diff_data", "LEG_H062_Wd", "Month", "Hour"]]

#-------------------------------
# prediction için zaman kaydırma
#-------------------------------
forecast_interval = 1
future_y = y_.shift(0)  
now_x = X_
X__ = now_x[:-forecast_interval]    
y__ = future_y[1:]
y__ = y__.apply(lambda x: 0 if x <= 0 else 1)

# Reset indices of X__ and y__ before boolean indexing
X__.reset_index(drop=True, inplace=True)
y__.reset_index(drop=True, inplace=True)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X__ = scaler.fit_transform(X__)


# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X__, y__, test_size=0.2, random_state=42)
# SMOTE ile veri çoğaltma
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Klasördeki tüm .joblib dosyalarını bulun
model_folder = '.'  # Çalışmakta olduğunuz klasör
model_files = [f for f in os.listdir(model_folder) if f.endswith('.joblib')]

# Sonuçları saklamak için bir liste
results = []

# Modelleri yükleyip SEDI, cr, fa, h, m değerlerini hesaplayın
for model_file in model_files:
    model_path = os.path.join(model_folder, model_file)
    loaded_model = load(model_path)
    
    test_predictions = loaded_model.predict(X_test)
    results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_test})
    
    cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
    fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
    h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
    m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()
    
    H = h / (h + m) if (h + m) != 0 else 0.01
    F = fa / (cr + fa) if (cr + fa) != 0 else 0.01
    
    SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))
    
    results.append({
        'Model': model_file,
        'SEDI': SEDI,
        'cr': cr,
        'fa': fa,
        'h': h,
        'm': m
    })

# Sonuçları bir DataFrame'e dönüştürün
results_df = pd.DataFrame(results)

# Sonuçları bir Excel dosyasına yazın
excel_path = 'model_results.xlsx'
results_df.to_excel(excel_path, index=False)

print(f"Excel dosyası oluşturuldu: '{excel_path}'")
