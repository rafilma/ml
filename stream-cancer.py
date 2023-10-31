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
Air_Pollution = st.number_input('Dalam skala 1-10 seberapa sering anda terpapar polusi udara ?', min_value=1.0, max_value=10.0, step=0.1)
Alcohol_use = st.number_input('Dalam skala 1-10 seberapa sering anda meminum alkohol ?', min_value=1.0, max_value=10.0, step=0.1)
Dust_Allergy = st.number_input('Dalam skala 1-10 apakah anda punya alergi terhadap debu ?', min_value=1.0, max_value=10.0, step=0.1)
OccuPational_Hazards = st.number_input('Dalam skala 1-10 apakah pekerjaan anda melibatkan bahan berbahaya ?', min_value=1.0, max_value=10.0, step=0.1)
Genetic_Risk = st.number_input('Dalam skala 1-10 apakah anda memiliki faktor keturunan yang beresiko ?', min_value=1.0, max_value=10.0, step=0.1)
chronic_Lung_Disease = st.number_input('Dalam skala 1-10 apakah anda punya riwayat penyakit paru-paru ?', min_value=1.0, max_value=10.0, step=0.1)
Balanced_Diet = st.number_input('Dalam skala 1-10 seberapa sering anda makan teratur ?', min_value=1.0, max_value=10.0, step=0.1)
Obesity = st.number_input('Dalam skala 1-10 apakah menurut anda anda sedang kelebihan berat badan ?', min_value=1.0, max_value=10.0, step=0.1)
Smoking = st.number_input('Dalam skala 1-10 seberapa sering anda merokok ?', min_value=1.0, max_value=10.0, step=0.1)
Passive_Smoker = st.number_input('Dalam skala 1-10 seberapa sering anda menjadi perokok Pasif ?', min_value=1.0, max_value=10.0, step=0.1)
Chest_Pain = st.number_input('Dalam skala 1-10 seberapa sering anda merasakan sakit di dada ?', min_value=1.0, max_value=10.0, step=0.1)
Coughing_of_Blood = st.number_input('Dalam skala 1-10 seberapa sering anda batuk berdarah ?', min_value=1.0, max_value=10.0, step=0.1)
Fatigue = st.number_input('Dalam skala 1-10 seberapa sering anda mengalami kelelahan ?', min_value=1.0, max_value=10.0, step=0.1)
Weight_Loss = st.number_input('Dalam skala 1-10 seberapa sering anda kekurangan berat badan ?', min_value=1.0, max_value=10.0, step=0.1)
Shortness_of_Breath = st.number_input('Dalam skala 1-10 seberapa sering anda mengalami sesak nafas ?', min_value=1.0, max_value=10.0, step=0.1)
Wheezing = st.number_input('Dalam skala 1-10 seberapa sering anda mengi pada pernafasan ?', min_value=1.0, max_value=10.0, step=0.1)
Swallowing_Difficulty = st.number_input('Dalam skala 1-10 seberapa sering anda mengalami kesulitan saat menelan ?', min_value=1.0, max_value=10.0, step=0.1)
Clubbing_of_Finger_Nails = st.number_input('Dalam skala 1-10 seberapa sering anda mengalami pembengkakan pada jari tangan ?', min_value=1.0, max_value=10.0, step=0.1)
Frequent_Cold = st.number_input('Dalam skala 1-10 seberapa sering anda terkena demam ?', min_value=1.0, max_value=10.0, step=0.1)
Dry_Cough = st.number_input('Dalam skala 1-10 seberapa sering anda batuk batuk ?', min_value=1.0, max_value=10.0, step=0.1)
Snoring = st.number_input('Dalam skala 1-10 seberapa sering anda mengorok saat tertidur ?', min_value=1.0, max_value=10.0, step=0.1)

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
