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

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
Variable di dasarkan pada data satu tahun terakhir
- Usia : Usia kita saat menjalakan test
- Jenis Kelamin : Jneis kelamin [1 jika kita adalah pria, 2 jika kita adalah wanita]
- Polusi Udara : Seberapa sering kita terpapar daerah yang rawan akan polusi udara
- Konsumsi Alkohol : Menunjukan seberapa sering kita mengkonsumsi alkohol
- Alergi terhadap debu : Menunjukan tingkat alergi kita terhadap debu
- Pekerjaan yang melibatkan bahan berbahaya : Menanyakan seberapa sering kita terpapar bahan berbahaya selama bekerja
- Faktor Genetik : Menanyakan faktor keturunan secara genetik apakah punya riwayat penyakit kanker paru
- Penyakit Paru Bawaan : Sudah sekronis apa penykit paru bawaan kita
- Makan Teratur : Seberapa sering kita makan secara teratur
- Kelebihan Berat Badan : Seberapa obesitas kah kita dalam skala 1-10 (berdasarkan kalkulasi berat badan ideal)
- Perokok : Seberapa sering kita merokok
- Perokok Pasif : Seberapa sering kita menjadi perokok pasif
- Nyeri Dada : Seberapa sering kita mengalami nyeri dada
- Batuk Berdarah : Seberapa sering kita mengalami batuk berdarah
- Kelelahan : Seberapa sering kita mengalami kelelahan (Kurang tidur, Bekerja lembur, overworked)
- Kekurangan Berat Badan : Seberapa sering kita kekurangan berart badan (Berdasarkan kalkulasi berat badan ideal)
- Sesak Nafas : Seberapa sering kita mengalami sesak dalam pernafasan
- Mengi pada Pernafasan : Seberapa sering kita mengalami mengi (wheezing) saat bernafas)
- Kesusahan dalam menelan : Seberapa sering kita mengalami kesusahan dalam menelan makan
- Pembengkakan pada Jari : Seberapa sering kita mengalami pembengkakan pada jari tangan
- Demam : Seberapa sering kita terkenan demam dan flu
- Batuk Batuk : Seberapa sering kita mengalami batuk batuk
- Ngorok : Seberapa sering kita mengorok saat tertidur


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


## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## Deployment
pada bagian ini anda memberikan link project yang diupload melalui streamlit share. boleh ditambahkan screen shoot halaman webnya.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

