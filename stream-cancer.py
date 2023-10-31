import pickle
import streamlit as st

diagnosis = pickle.load(open('resiko_cancer.sav', 'rb'))

st.title('Diagnosa Resiko Anda Terkena Penyakit Kanker Paru')

Age = st.text_input('Usia Anda')
Gender = st.selectbox('Jenis Kelamin anda ?', ['Laki laki', 'Wanita'])

if Gender == 'Laki laki':
    Gender = 1
else:
    Gender = 2
Air_Pollution = st.text_input('Dalam skala 1-10 seberapa sering anda terpapar polusi udara ?', min_value=1.0, max_value=10.0, step=0.1)
Alcohol_use = st.text_input('Dalam skala 1-10 seberapa sering anda meminum alkohol ?')
Dust_Allergy = st.text_input('Dalam skala 1-10 apakah anda punya alergi terhadap debu ?')
OccuPational_Hazards = st.text_input('Dalam skala 1-10 apakah pekerjaan anda melibatkan bahan berbahaya ?')
Genetic_Risk = st.text_input('Dalam skala 1-10 apakah anda memiliki faktor keturunan yang beresiko ?')
chronic_Lung_Disease = st.text_input('Dalam skala 1-10 apakah anda punya riwayat penyakit paru-paru ?')
Balanced_Diet = st.text_input('Dalam skala 1-10 seberapa sering anda makan teratur ?')
Obesity = st.text_input('Dalam skala 1-10 apakah menurut anda anda sedang kelebihan berat badan ?')
Smoking = st.text_input('Dalam skala 1-10 seberapa sering anda merokok ?')
Passive_Smoker = st.text_input('Dalam skala 1-10 seberapa sering anda menjadi perokok Pasif ?')
Chest_Pain = st.text_input('Dalam skala 1-10 seberapa sering anda merasakan sakit di dada ?')
Coughing_of_Blood = st.text_input('Dalam skala 1-10 seberapa sering anda batuk berdarah ?')
Fatigue = st.text_input('Dalam skala 1-10 seberapa sering anda mengalami kelelahan ?')
Weight_Loss = st.text_input('Dalam skala 1-10 seberapa sering anda kekurangan berat badan ?')
Shortness_of_Breath = st.text_input('Dalam skala 1-10 seberapa sering anda mengalami sesak nafas ?')
Wheezing = st.text_input('Dalam skala 1-10 seberapa sering anda mengi pada pernafasan ?')
Swallowing_Difficulty = st.text_input('Dalam skala 1-10 seberapa sering anda mengalami kesulitan saat menelan ?')
Clubbing_of_Finger_Nails = st.text_input('Dalam skala 1-10 seberapa sering anda mengalami pembengkakan pada jari tangan ?')
Frequent_Cold = st.text_input('Dalam skala 1-10 seberapa sering anda terkena demam ?')
Dry_Cough = st.text_input('Dalam skala 1-10 seberapa sering anda batuk batuk ?')
Snoring = st.text_input('Dalam skala 1-10 seberapa sering anda mengorok saat tertidur ?')

cancer_risk = ''

if st.button('Diagnosa'):
    cancer_predict = diagnosis.predict([[Age, Gender, Air_Pollution, Alcohol_use, Dust_Allergy, OccuPational_Hazards, Genetic_Risk, chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking, Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue, Weight_Loss, Shortness_of_Breath, Wheezing, Swallowing_Difficulty, Clubbing_of_Finger_Nails, Frequent_Cold, Dry_Cough, Snoring]])
    
    if(cancer_predict[0] == 1):
        cancer_risk = 'Anda Beresiko Rendah Terkena Kanker Paru'
    elif(cancer_predict[0] == 2):
        cancer_risk = 'Anda Beresiko Menengah Terkena Kanker Paru'
    else :
        cancer_risk ='Anda Beresiko Tinggi Terkenan Kanker Paru'

    st.success(cancer_risk)
