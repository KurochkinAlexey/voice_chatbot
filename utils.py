import pandas as pd
import numpy as np
import sqlite3
import subprocess


names = ['James', 'David', 'Christopher', 'George', 'Ronald', 'John', 'Richard', 'Daniel', 'Kenneth', 'Anthony',
    'Robert', 'Charles', 'Michael', 'Jospeh', 'Mark', 'Edward'] + ['Mary', 'Jennifer', 'Lisa', 'Sandra', 'Michelle', 'Patricia', 'Maria', 'Nancy',
                           'Donna', 'Laura', 'Linda', 'Susan', 'Karen', 'Barbara', 'Margaret', 'Betty', 'Ruth', 'Kimberly']
surnames = ['Smith', 'Anderson', 'Clark', 'Wright', 'Mitchell', 'Johnson', 'Thomas', 'Rodriguez', 'Lopez', 'Williams',
           'Jackson', 'Lewis', 'Hill', 'Roberts', 'Jones', 'White', 'Lee', 'Scott', 'Turner', 'Davis', 'Martin', 'Hall',
           'Adams', 'Campbell']

specs = ['General practitioner', 'Cardiologist', 'Gynecologist', 'Oncologist', 'Surgeon', 'Dermatologist', 'Gastroenterologist',
        'Ophthalmologist', 'Pediatrician']

interests = ['Cardiology', 'Cardiothoracic surgery', 'Child and adolescent psychiatry', 'Clinical neurophysiology',
            'Allergy and immunology', 'Clinical neurophysiology', 'Dermatology', 'Hematology', 'Emergency medicine']

institutions = ['Hospital', 'Clinic', 'Nursing home', 'Birth center', 'Hospice care center', 'Radiology center']
public = [True, False, False, True, False, True]
address = ['Address {}'.format(i) for i in range(6)]

def generate_db(max_tries=1000, n_samples=100, db_name='medical.db'):
    
    for i in range(max_tries):
        N_SAMPLES = n_samples
        np.random.seed(i)
        all_names = np.random.choice(names, size=N_SAMPLES)
        all_surnames = np.random.choice(surnames, size=N_SAMPLES)
        full_names = [s1+" "+s2 for s1, s2 in zip(all_names, all_surnames)]
        all_specs = np.random.choice(specs, size=N_SAMPLES)
        all_interests = np.random.choice(interests, size=N_SAMPLES)
        all_institutions = np.random.choice(institutions, size=N_SAMPLES)
        if len(full_names) == len(set(full_names)): #to avoid full_name duplicates
            break
    
    df_doctors = pd.DataFrame({
        'full_name': full_names,
        'specialization': all_specs,
        'field_of_interest': all_interests,
        'institution': all_institutions
    })
    
    df_inst = pd.DataFrame({
        'institution': institutions,
        'public': public,
        'address': address,
    })
    
    subprocess.call(['sqlite3', db_name, "VACUUM;"])
    
    conn = sqlite3.connect(db_name)

    df_doctors.to_sql('Doctors', conn, if_exists='replace', index=False)
    df_inst.to_sql('Institutions', conn, if_exists='replace', index=False)

    conn.close()
    
    
def get_medical_keywords():
    common = ['patient', 'diagnosis', 'treatment', 'hospital', 'medicine', 'doctor', 'doctors', 'institutions',
             'database', 'data', 'table'] + specs + interests + institutions
    result = set([w2.lower() for w1 in common for w2 in w1.split(' ')]) - {'general', 'center', 'practicioner'}
    return list(result)