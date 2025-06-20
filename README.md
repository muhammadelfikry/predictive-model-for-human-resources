# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Maju merupakan perusahaan multinasional yang telah berdiri sejak tahun 2000 dan memiliki lebih dari 1000 karyawan yang tersebar di seluruh Indonesia. Walaupun telah berkembang menjadi perusahaan besar, Jaya Jaya Maju menghadapi tantangan dalam mengelola karyawannya, yang berdampak pada tingginya tingkat attrition (keluar masuk karyawan), yaitu lebih dari 10%.

Manajemen menyadari bahwa tingginya angka attrition dapat mengganggu stabilitas operasional dan budaya perusahaan. Oleh karena itu, manajer departemen HR ingin melakukan analisis mendalam terhadap faktor-faktor yang memengaruhi tingkat attrition dan mengembangkan dashboard interaktif untuk memantau faktor-faktor tersebut secara real-time.

### Permasalahan Bisnis

- Mengapa tingkat attrition di Jaya Jaya Maju cukup tinggi?
- Faktor apa saja yang berkontribusi terhadap keputusan karyawan untuk keluar?
- Bagaimana cara memantau faktor-faktor tersebut secara efisien dan informatif?

### Cakupan Proyek

- Melakukan analisis data untuk mengidentifikasi faktor-faktor yang mempengaruhi attrition.
- Mengembangkan model machine learning untuk memprediksi kemungkinan karyawan keluar.
- Menerapkan proses preprocessing dan prediksi otomatis menggunakan Python.
- Membangun business dashboard interaktif yang menampilkan visualisasi insight dari data attrition.
- Menyimpan dan menampilkan data attrition yang telah diprediksi melalui integrasi dengan database.

### Persiapan

Sumber data: 
https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv

Setup environment:

Jalankan kode berikut untuk melakukan installasi library yang dibutuhkan untuk menjalankan proyek ini. 

```bash
pip install requirements.txt
```

Cara penggunaan model prediksi:
```python
import pandas as pd
import joblib

def preprocessing(df):
    selected_features = ["WorkLifeBalance", "JobSatisfaction", "JobLevel", "MonthlyIncome",
                         "Age", "MaritalStatus", "Department", "OverTime", "Attrition"]
    
    df = df[selected_features]
    df = df.drop("Attrition", axis=1)

    categorical_features = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_features)

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = joblib.load("scaler.pkl")
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

def model_predict(df):
    df = preprocessing(df)

    model = joblib.load("model.pkl")
    predictions = model.predict(df)
    df["Attrition"] = predictions

    return df[["Attrition"]]

def main():
    input_data = pd.read_csv("https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv")
    input_data = input_data[input_data["Attrition"].isna()].copy()
    predictions = model_predict(input_data)

    return predictions

if __name__ == "__main__":
    result = main()
    print(result)
```

Dataset dapat diganti pada ```input_data``` yang ada pada fungsi ```def main```. Namun pastikan dataset yang digunakan memiliki fitur ```["WorkLifeBalance", "JobSatisfaction", "JobLevel", "MonthlyIncome", "Age", "MaritalStatus", "Department", "OverTime", "Attrition"]```

Jalankan perintah berikut pada terminal untuk melakukan prediksi:
```bash
python prediction.py
```

## Business Dashboard

Dashboard yang dikembangkan bertujuan untuk membantu manajer HR dalam memahami dan memantau kondisi attrition di perusahaan secara visual dan interaktif. Dashboard ini menyajikan:

- Komposisi total karyawan yang stay vs attrition.

- Distribusi attrition berdasarkan overtime, job level, dan department.

- Rata-rata gaji bulanan berdasarkan status attrition.

- Insight visual bahwa Job Level 1 dan departemen Research & Development memiliki tingkat attrition tertinggi.

Dashboard ini terhubung langsung dengan database yang menyimpan hasil prediksi dan data aktual sehingga dapat diperbarui secara dinamis.

**Metabase**:

![muhammadelfikry-dashboard](https://github.com/user-attachments/assets/97215261-eb01-4fca-a785-5be0d81695ba)

**Menjalankan dashboard**:

Buka **Command Prompt (CMD)** di direktori tempat folder `metabase-data` berada, lalu jalankan perintah berikut. pastikan docker sudah terinstall:

```cmd
docker run -d -p 3000:3000 ^
  -v %cd%\metabase-data:/metabase.db ^
  -e MB_DB_TYPE=h2 ^
  -e MB_DB_FILE=/metabase.db/metabase.db ^
  --name metabase ^
  metabase/metabase
```

Setelah container berjalan, buka browser dan akses:

```bash
http://localhost:3000/setup
```

Pada halaman utama Metabase masukan email dan password berikut untuk login:

**email**: root@mail.com

**password**: root123

## Conclusion

Berdasarkan hasil analisis dan visualisasi:

- Karyawan yang bekerja lembur (overtime) memiliki kemungkinan keluar lebih tinggi.

- Level pekerjaan rendah (Job Level 1) memiliki tingkat keluar paling tinggi.

- Departemen Research & Development dan Sales paling banyak mengalami attrition.

- Karyawan yang keluar memiliki pendapatan bulanan yang lebih rendah dibandingkan yang tetap.

- Model prediksi yang dikembangkan dapat digunakan untuk memantau potensi attrition di masa depan, dan dashboard membantu dalam mengambil keputusan strategis secara cepat dan informatif.

### Rekomendasi Action Items

Berikut beberapa rekomendasi action items yang dapat digunakan guna mencapai target spesifik:

- Department: Berguna untuk melihat visualisasi berdasarkan departemen.
- Job Level: Berguna untuk melihat visualiasi berdasarkan Job Level.
- Is Work Over Time?: Berguna untuk melihat visualisasi berdasarkan apakah karyawan bekerja dari waktu ke waktu.
