import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from joblib import dump, load
import math
import numpy as np

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


param_grid = {
    'n_estimators': [5, 10, 25, 50, 100, 200, 300],
    'max_depth': [None,2, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

best_auc_score = -np.inf  # En yüksek AUC skoru
best_auc_params = None
best_auc_model = None


for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                # RandomForestClassifier'ı oluştur
                rf = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            random_state=42)
                # Modeli eğit
                rf.fit(X_train_resampled, y_train_resampled)
                # Test seti üzerinde tahmin yap
                test_probabilities = rf.predict_proba(X_test)[:, 1]
                # AUC değerini hesapla
                auc_score = roc_auc_score(y_test, test_probabilities)
                # Eğer bu modelin AUC skoru en iyi skordan daha iyiyse, en iyi skoru ve modeli güncelle
                if auc_score > best_auc_score:
                    best_auc_score = auc_score
                    best_auc_params = {'n_estimators': n_estimators,
                                       'max_depth': max_depth,
                                       'min_samples_split': min_samples_split,
                                       'min_samples_leaf': min_samples_leaf}
                    best_auc_model = rf  # En iyi modeli güncelle

                predictions = rf.predict(X_test)
                test_probabilities = rf.predict_proba(X_test)[:, 1]
                results_df = pd.DataFrame({'y_test': y_test, 'Prediction': predictions})
                # SEDI hesaplama
                cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
                fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
                h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
                m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()

                H = h / (h + m) if (h + m) != 0 else 0
                F = fa / (cr + fa) if (cr + fa) != 0 else 0

                if F == 0:
                    F = 0.01
                if H == 0:
                    H = 0.01

                SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (
                            math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))
                
                # Eğer AUC skoru 0.8'den büyükse, modeli kaydet
                if SEDI > 0.7:
                    if auc_score > 0.8:
                        model_filename = f'auc_model_s{SEDI}_n{n_estimators}_d{max_depth}_s{min_samples_split}_l{min_samples_leaf}.joblib'
                        dump(rf, model_filename)
                        print(f"AUC {auc_score:.2f} olan model '{model_filename}' dosyasına kaydedildi.")

# En iyi modelin kaydedilmesi
if best_auc_model is not None:
    dump(best_auc_model, 'best_model_auc.joblib')
    print(f"Optimize edilmiş model 'best_model_auc.joblib' dosyasına kaydedildi.")

"""

# Modeli yükleme
loaded_model = load('best_model_auc.joblib')
test_predictions = loaded_model.predict(X_test)
results_df = pd.DataFrame({'Prediction': test_predictions, 'y_test': y_test})

#-------------------------------------------------------------------------------------


# y_test'in 1'e eşit olduğu durumlarda predictions'ın da 1'e eşit olduğu durumları sayma
matching_true_alarms = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 1)])
matching_false_no_alarms = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 1)])
matching_true_no_alarm = len(results_df[(results_df['y_test'] == 0) & (results_df['Prediction'] == 0)])
matching_false_alarm = len(results_df[(results_df['y_test'] == 1) & (results_df['Prediction'] == 0)])

# Sonuçları CSV dosyasına yazma
results_df.to_csv('predictions_vs_y_test.csv', index=False)

print("CSV dosyası oluşturuldu: 'predictions_vs_y_test.csv'", X_train.shape, X_test.shape)
print("matching_catch_alarms:", matching_true_alarms)
print("matching_false_no_alarms:", matching_false_no_alarms)
print("matching_catch_no_alarm", matching_true_no_alarm)
print("matching_false_alarm", matching_false_alarm)

# SEDI hesaplama
cr = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 0)).sum()
fa = ((results_df['y_test'] == 0) & (results_df['Prediction'] == 1)).sum()
h = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 1)).sum()
m = ((results_df['y_test'] == 1) & (results_df['Prediction'] == 0)).sum()

H = h / (h + m)
F = fa / (cr + fa)

if F == 0:
    F = 0.01
if H == 0:
    H = 0.01

SEDI = (math.log(F) - math.log(H) - math.log(1 - F) + math.log(1 - H)) / (math.log(F) + math.log(H) + math.log(1 - F) + math.log(1 - H))

print("SEDI:", SEDI)


"""












