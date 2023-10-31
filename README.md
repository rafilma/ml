# Laporan Proyek Machine Learning
### Nama : Rafil Moehamad Alif
### Nim : 211351116
### Kelas : Malam B

## Domain Proyek

Proyek yang saya angkat kali ini adalah kalkulasi resiko manusia terkena kanker paru paru beradasar 23 parimeter yang di tentukan, dimana menurut penelitian dalam 1.59 Juta korban kanker mayoritas di pengaruhi oleh faktor luar seperti asap rokok dan polusi udara.
Maka dari itu, saya selaku pembuat mencoba mengkalkulasikan tingkat resiko anda terkena kanker paru untuk menjadikan sebagai tindakan pencegahan guna anda menjadi lebih waspada


## Business Understanding

Proyek ini memudahkan kita untuk melakukan tindakan pencegahan jika kita terbukti rawan dan beresiko tinggi terpapar kanker paru menggunakan algoritma klasifikasi SVC

Bagian laporan ini mencakup:

### Problem Statements

Seseorang bisa saja terkenan kanker paru berdasarkan parimeter :
- Usia
- Jenis Kelamin
- Polusi Udara
- Konsumsi Alkohol
- Alergi terhadap debu
- Pekerjaan yang melibatkan bahan berbahaya
- Faktor Genetik
- Penyakit Paru Bawaan
- Makan Teratur
- Kelebihan Berat Badan
- Perokok
- Perokok Pasif
- Nyeri Dada
- Batuk Berdarah
- Kelelahan
- Kekurangan Berat Badan
- Sesak Nafas
- Mengi pada Pernafasan
- Kesusahan dalam menelan
- Pembengkakan pada Jari
- Demam
- Batuk Batuk
- Ngorok

Jenis Penelitian yang digunakan adalah menggunakan skala 1-10 berdasarkan parimeter diatas, dan hasilnya kemudian di kalkulasikan apakah kita tergolong rendah resiko atau tinggi resiko

### Goals

Agar kita lebih waspada dalam menjaga pola hidup kita supaya terhindar dari resiko kanker Paru

### Solution Statements
- Dikembangan sebagai kalkukalasi tingkat resiko kita terkena kanker paru paru berdasarkan faktor eksternal dan internal
- Menjadi bahan untuk meningkatkan kewaspadaan terhadap resiko terkena kanker paru paru supaya bisa di lakukan tindakan pencegahan

## Data Understanding
Dataset yang dipergunakan dalan proyek ini di dapatkan di kaggle dimana dataset ini di dasarkan padan 1000 studi kasus terhadap pasien kanker dan di teliti berdasarkan 23 attribut diatas.
Dataset tersebut sejatinya dibuat untuk mempelajari dan memprediksi faktor apa saja yang menjadi pendukung seseorang terkena kanker paru paru dan solusi apa saja yang bisa di lakukan sebagai tindakan pencegahan.

[Lung Cancer Prediction](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link).

Variable yang di pergunakan sebagai parimeter meliputi :  

### Variabel-variabel yang digunakan adalah sebagai berikut:
Variable di dasarkan pada data satu tahun terakhir dimana setiap parameternya di isikan data integer untuk skala 1-10
- Usia : Usia kita saat menjalakan test (integer)
- Jenis Kelamin : Jeniss kelamin (integer)
- Polusi Udara : Seberapa sering kita terpapar daerah yang rawan akan polusi udara (integer)
- Konsumsi Alkohol : Menunjukan seberapa sering kita mengkonsumsi alkohol (integer)
- Alergi terhadap debu : Menunjukan tingkat alergi kita terhadap debu (integer)
- Pekerjaan yang melibatkan bahan berbahaya : Menanyakan seberapa sering kita terpapar bahan berbahaya selama bekerja (integer)
- Faktor Genetik : Menanyakan faktor keturunan secara genetik apakah punya riwayat penyakit kanker paru (integer)
- Penyakit Paru Bawaan : Sudah sekronis apa penykit paru bawaan kita (integer)
- Makan Teratur : Seberapa sering kita makan secara teratur (integer)
- Kelebihan Berat Badan : Seberapa obesitas kah kita dalam skala 1-10 (berdasarkan kalkulasi berat badan ideal) (integer)
- Perokok : Seberapa sering kita merokok (integer)
- Perokok Pasif : Seberapa sering kita menjadi perokok pasif (integer)
- Nyeri Dada : Seberapa sering kita mengalami nyeri dada (integer)
- Batuk Berdarah : Seberapa sering kita mengalami batuk berdarah (integer)
- Kelelahan : Seberapa sering kita mengalami kelelahan (Kurang tidur, Bekerja lembur, overworked) (integer)
- Kekurangan Berat Badan : Seberapa sering kita kekurangan berart badan (Berdasarkan kalkulasi berat badan ideal) (integer)
- Sesak Nafas : Seberapa sering kita mengalami sesak dalam pernafasan (integer)
- Mengi pada Pernafasan : Seberapa sering kita mengalami mengi (wheezing) saat bernafas) (integer)
- Kesusahan dalam menelan : Seberapa sering kita mengalami kesusahan dalam menelan makan (integer)
- Pembengkakan pada Jari : Seberapa sering kita mengalami pembengkakan pada jari tangan (integer)
- Demam : Seberapa sering kita terkenan demam dan flu (integer)
- Batuk Batuk : Seberapa sering kita mengalami batuk batuk (integer)
- Ngorok : Seberapa sering kita mengorok saat tertidur (integer)


## Data Preparation
Data yang diambil berdaasrkan data kaggle

Pertama kita import dulu library yang di butuh dengan memasukan perintah :

```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
```
Kemudian agar aplikasi bisa berjalan otomatis di collab maka kita harus mengkoneksikan file token kaggle kedalam aplikasi kita dan membuat directory khusus.

```bash
from google.colab import files
files.upload()
```
upload file token kaggle kita kemudian

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Jika direktori sudah dibuat maka kita bisa mendownload datasetnya 

```bash
!kaggle datasets download -d thedevastator/cancer-patients-and-air-pollution-a-new-link
```

Setelah terdownload, kita bisa mengekstrak dataset terserbut dan memindahkan nya kedalam folder yang sudah di buat 

```bash
!mkdir cancer_data
!unzip cancer-patients-and-air-pollution-a-new-link.zip -d cancer_data
!ls cancer_data
```

Jika sudah, maka kita bisa langsung membuka file dataset tersebut

```bash
df = pd.read_csv('cancer_data/cancer patient data sets.csv')
```
 kemudian kita bisa panggil data tersebut, karena saya definisakan dengan df maka saya bisa panggil dengan cara

 ```bash
df.head()
```
Selanjutnya kita bisa lihat kolom apa saja yang terdata di dataset tersebut dengan perintah

```bash
df.columns
```
kemudian jika sudah tampil nama nama kolomnya kita bisa eliminasi beberapa kolom dan hanya menampilkan kolom yang di inginkan dengan perintah

```bash
df=df[['Age', 'Gender', 'Air Pollution', 'Alcohol use',
       'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
       'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
       'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue',
       'Weight Loss', 'Shortness of Breath', 'Wheezing',
       'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold',
       'Dry Cough', 'Snoring', 'Level']]
```
Maka Kolom yang ditampilkan hanya kolom yang sudah di definisikan diatas. Tapi karena kita akan membuat sebuat model pembelajaran mesin dimana kolom tersebut akan menjadi variable yangmana tidak boleh ada spasi, maka kita bisa merubah nama kolom dengan perintah berikut

```bash
colname = ['Age', 'Gender', 'Air_Pollution', 'Alcohol_use',
       'Dust_Allergy', 'OccuPational_Hazards', 'Genetic_Risk',
       'chronic_Lung_Disease', 'Balanced_Diet', 'Obesity', 'Smoking',
       'Passive_Smoker', 'Chest_Pain', 'Coughing_of_Blood', 'Fatigue',
       'Weight_Loss', 'Shortness_of_Breath', 'Wheezing',
       'Swallowing_Difficulty', 'Clubbing_of_Finger_Nails', 'Frequent_Cold',
       'Dry_Cough', 'Snoring', 'Level']
df.columns = colname
df
```
Kemudian karena model ini hanya bisa menggunakan inputan integer maka kita harus rubah value dari kolom diagnosis dengan perintah

```bash
df=df.replace({'Level':{'Low': 1, 'Medium': 2, 'High': 3}})
```
Maka data untuk kolom 'Level' akan berubah menjadi angka.


**Visualization**
Disini kita coba lakukan sebuuah visualisasi data dengan modul seaborn
```bash
import seaborn as sns
sns.set()
```
![alt text](https://github.com/rafilma/ml/blob/main/eda1.png)

Atau kita bisa coba visualisasikan dengan cara lain yakni
```bash
sns.heatmap(df.corr()[['Level']].sort_values(by='Level', ascending=False), vmin=-1, vmax=1, annot=True, cmap='GnBu')
```
Dan akan diperoleh heatmap berikut
![alt text](https://github.com/rafilma/ml/blob/main/eda2.png)

Selanjutnya kita bisa buat distribusi dengan pie chart
```bash
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].pie(df['Level'].value_counts(), labels=df['Level'].value_counts().index, autopct='%.0f%%')
axs[0].set_title('Distribution by Level')

axs[1].pie(df['Gender'].value_counts(), labels=df['Gender'].value_counts().index, autopct='%.0f%%')
axs[1].set_title('Distribution by Gender')

plt.tight_layout()
plt.show()
```
![pie](https://github.com/rafilma/ml/assets/148635738/f4c93e15-ca7f-4691-9f9d-0aab5f1287c7)


## Modeling
Untuk melakukan proses modeling, disini saya menggunakan klasifikasi dengan algoritma SVC.
Pertama kita harus menentukan parimeter X dan Y. X merupakan atrribut dan Y adalah Label
karena kita akan menggunakan semua attribut dan mengecualikan satu kolom saja sebagai label maka bisa kita ketikan perintah
```bash
X = df.drop (columns='Level', axis=1)
Y = df['Level']
```
Maka X akan menjadi semua kolom kecuali Level

Selanjutnya kita lakukan standarisi data dengan perintah

```bash
scaler = StandardScaler()
```
```bash
scaler.fit(X)
```
```bash
standarized_data = scaler.transform(X)
```
```bash
X = standarized_data
Y = df['Level']
```
Maka data yang terdapat pada nilai X akan di standarkan

Selanjutkan kita lakukan pengetesan akurasi data

```bash
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify=Y, random_state=2)
```
Data test akan di jalankan sebanyak 30% dan data train sebanyak 70%

Selanjutnya kita masukan dulu perintah klasifikasinya

```bash
classifier = svm.SVC(kernel='linear')
```
```bash
classifier.fit(X_train, Y_train)
```
Jika sudah dimasukan maka kita bisa lakukan cek nilai akurasi data training
```bash
x_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(x_train_prediction, Y_train)
```
```bash
print('Tingkat akurasi data training = ', training_data_accuracy)
```
```bash
Tingkat akurasi data training =  1.0
```
Didapat nilai akurasi 100%, hal ini dipengaruhi oleh jumlah parameter pada data yang digunakan, jika parameternya dikurangi maka tingkat akurasi juga terpengaruh

Selanjutnya bisa dilakukan pengetesan dengan nilai array seperti kolom pada dataset
```bash
input_data = (17, 1, 3, 1, 5, 3, 4, 2, 2, 2, 2, 4, 2, 3, 1, 3, 7, 8, 6, 2, 1, 7, 2)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 1):
    print('Rendah Resiko')
elif (prediction[0] == 2):
        print('Resiko Menengah')
else :
    print('Resiko Tinggi')
```
Maka akan di peroleh ouput :
```bash
[[-1.68123833 -0.81990292 -0.41391868 -1.36035665 -0.08333998 -0.87338274
  -0.27282112 -1.28816247 -1.16703997 -1.16062345 -0.78086997 -0.08439285
  -1.069735   -0.7660449  -1.27301449 -0.38767737  1.20843596  2.06918561
   0.99328083 -0.80566309 -1.38459305  1.54417079 -0.6282445 ]]
[2]
Resiko Menengah
```
Berdasarkan inputan kita, maka bisa di ketahui bahwa tingkat resiko kita terkena kanker paru paru sejauh mana


## Evaluation
Pada tabel evaluasi kita tentukan nilai f1 score dengan nilai presisinya menggunakan tabel matrix confusion
maka didapatkan matrix sebagai berikut

![alt text](https://github.com/rafilma/ml/blob/main/matrix%20evaluation.png)

dan ketika di jabarkan kita akan mendapatkan score :
![alt text](https://github.com/rafilma/ml/blob/main/score%20f1.png)

Bisa disimpulkan Akurasinya sebanyak 100% dengan rata rata 99.5%
Aplikasi ini sangat bisa digunakan untuk salah satu cara melakukan diagnosa awal apakah kita rawan terkena resiko kanker paru paru atau tidak supaya bisa dilakukan tindakan pencegahan jauh jauh hari.


## Deployment
[Diagnosa Resiko Terkena Kanker Paru](https://utsmachinelearning.streamlit.app/)
![alt text](https://github.com/rafilma/ml/blob/main/Screenshot%20(24).png)
