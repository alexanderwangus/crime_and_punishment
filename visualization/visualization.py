import csv
import numpy as np

def overview():
    print('=== Overview ===') 
    print('Number of people in data set: ', 7214)

def age():
    AGE_COL = 14
    num_people = 7214
    age_group = np.zeros((3))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            age_group[(int)(row[AGE_COL])] += 1
    age_group = age_group.astype(int)
    print()
    print('=== AGE ===')
    print('Below 25:', age_group[0], '(', round(100 * (float)(age_group[0]) / num_people), '%)')
    print('Between 25-45:', age_group[1], '(', round(100 * (float)(age_group[1]) / num_people), '%)')
    print('Above 45:', age_group[2], '(', round(100 * (float)(age_group[2]) / num_people), '%)')

def race():
    RACE_COL = 6
    num_people = 7214
    race_breakdown = np.zeros((6))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            for i in range(RACE_COL):
                if (row[30 + i] == '1'):
                    race_breakdown[i] += 1
    race_breakdown = race_breakdown.astype(int)
    print()
    print('=== RACE ===')
    print('Caucasian:', race_breakdown[2], '(', round(100 * (float)(race_breakdown[2]) / num_people), '%)')
    print('African American:', race_breakdown[0], '(', round(100 * (float)(race_breakdown[0]) / num_people), '%)')
    print('Hispanic:', race_breakdown[3], '(', round(100 * (float)(race_breakdown[3]) / num_people), '%)')
    print('Asian:', race_breakdown[1], '( %.3f'%(100 * (float)(race_breakdown[1]) / num_people), '%)')
    print('Native American:', race_breakdown[4], '( %.3f'%(100 * (float)(race_breakdown[4]) / num_people), '%)')
    print('Other:', race_breakdown[5], '( %.3f'%(100 * (float)(race_breakdown[5]) / num_people), '%)')

def gender():
    GENDER_COL = 18
    num_people = 7214
    gender_breakdown = np.zeros((2))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            gender_breakdown[(int)(row[GENDER_COL])] += 1
    gender_breakdown = gender_breakdown.astype(int)
    print()
    print('=== GENDER ===')
    print('Male:', gender_breakdown[1], '(', round(100 * (float)(gender_breakdown[1]) / num_people), '%)')
    print('Female:', gender_breakdown[0], '(', round(100 * (float)(gender_breakdown[0]) / num_people), '%)') 

def overall_recid():
    OVERALL_RECID_COL = 11
    num_people = 7214
    recid = np.zeros((2))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            recid[(int)(row[OVERALL_RECID_COL])] += 1
    recid = recid.astype(int)
    print()
    print('=== OVERALL RECIDIVISM ===')
    print('Recidivists:', recid[1] , '(', round(100 * (float)(recid[1]) / num_people), '%)')
    print('Non-recidivists:', recid[0], '(', round(100 * (float)(recid[0]) / num_people), '%)') 

def two_year_recid():
    TWO_YEAR_RECID_COL = 37
    num_people = 7214
    recid = np.zeros((2))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            recid[(int)(row[TWO_YEAR_RECID_COL])] += 1
    recid = recid.astype(int)
    print()
    print('=== TWO-YEAR RECIDIVISM ===')
    print('Two-year recidivists:', recid[1] , '(', round(100 * (float)(recid[1]) / num_people), '%)')
    print('Non-two-year recidivists:', recid[0], '(', round(100 * (float)(recid[0]) / num_people), '%)') 

def charge_degree():
    CHARGE_DEGREE_COL = 24
    num_people = 7214
    charge_degree = np.zeros((2))
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            charge_degree[(int)(row[CHARGE_DEGREE_COL])] += 1
    charge_degree = charge_degree.astype(int)
    print()
    print('=== CHARGE DEGREE ===')
    print('Misdemeanor:', charge_degree[0] , '(', round(100 * (float)(charge_degree[0]) / num_people), '%)')
    print('Felony:', charge_degree[1], '(', round(100 * (float)(charge_degree[1]) / num_people), '%)') 

def recid_violent():
    VIOLENT_RECID_COL = 7
    OVERALL_RECID_COL = 11
    num_people = 7214
    recid_violent = np.zeros((2))
    num_recid = 0
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[OVERALL_RECID_COL] == '1':
               num_recid += 1
               recid_violent[(int)(row[VIOLENT_RECID_COL])] += 1
    recid_violent = recid_violent.astype(int)
    print()
    print('=== VIOLENCE AMONG RECIDIVISTS ===')
    print('Total # or recidivists: ', num_recid)
    print('Recidivists who were violent:', recid_violent[1] , '(', round(100 * (float)(recid_violent[1] / num_recid)), '%)')
    print('Recidivists who were nonviolent:', recid_violent[0], '(', round(100 * (float)(recid_violent[0] / num_recid)), '%)')

def recid_prison_duration():
    DURATION_COL = 10
    OVERALL_RECID_COL = 11
    num_people = 7214
    duration = np.zeros((num_people))
    recid_duration = np.zeros((2))
    num_recid = 0
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        i = 0
        for row in reader:
            if row[DURATION_COL] == '':
                duration[i] = 0
                i += 1
                continue
            num_recid += 1
            duration[i] = int(row[DURATION_COL])
            recid_duration[(int)(row[OVERALL_RECID_COL])] += duration[i]
            i += 1
        mu = np.sum(duration) / num_recid 
        std_dev = np.std(duration)
    print()
    print('=== PRISON DURATION AMONG RECIDIVISTS ===')
    print('NOTE: Some outliers in duration skewed metrics (i.e. several people were imprisoned for 40 000+ days, whereas most were imprisoned for several hundred days')
    print('NOTE: Sentence duration data unavailable for non-recidivists')
    print('Mean: %.3f'%mu)
    print('Standard deviation:%.3f'%std_dev)

print()
overview()
age()
race()
gender()
overall_recid()
two_year_recid()
charge_degree()
recid_prison_duration()
recid_violent()
print()
