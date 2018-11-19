import csv
import numpy as np
import matplotlib.pyplot as plt

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
    percentages = [round(100 * (float)(age_group[0]) / num_people), round(100 * (float)(age_group[1]) / num_people), round(100 * (float)(age_group[2]) / num_people)] 
    print('Below 25:', age_group[0], '(', percentages[0], '%)')
    print('Between 25-45:', age_group[1], '(', percentages[1], '%)')
    print('Above 45:', age_group[2], '(', percentages[2], '%)')
    x_range = range(len(percentages))
    width = 1 / 1.5 
    x = ['Below 25', '25-45', 'Above 45']
    plt.title("Age Breakdown")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Age Group')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(race) / num_people) for race in race_breakdown]
    print()
    print('=== RACE ===')
    print('African American:', race_breakdown[0], '( %.3f'%percentages[0], '%)')
    print('Asian:', race_breakdown[1], '( %.3f'%percentages[1], '%)')
    print('Caucasian:', race_breakdown[2], '( %.3f'%percentages[2], '%)')
    print('Hispanic', race_breakdown[3], '( %.3f'%percentages[3], '%)')
    print('Native American:', race_breakdown[4], '( %.3f'%percentages[4], '%)')
    print('Other:', race_breakdown[5], '( %.3f'%percentages[5], '%)')
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['African American', 'Asian', 'Caucasian', 'Hispanic', 'Native American', 'Other']
    plt.title("Race Breakdown")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Race')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(gender) / num_people) for gender in gender_breakdown] 
    print()
    print('=== GENDER ===')
    print('Male:', gender_breakdown[1], '(', percentages[1], '%)')
    print('Female:', gender_breakdown[0], '(', percentages[0], '%)') 
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['Female', 'Male']
    plt.title("Gender Breakdown")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Gender')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(did_recid) / num_people) for did_recid in recid] 
    print()
    print('=== OVERALL RECIDIVISM ===')
    print('Recidivists:', recid[1] , '(', percentages[1], '%)')
    print('Non-recidivists:', recid[0], '(', percentages[0], '%)') 
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['Did not recidivize', 'Recidivized']
    plt.title("Recidivism Outcome")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Recidivism Outcome')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(did_recid) / num_people) for did_recid in recid] 
    print()
    print('=== TWO-YEAR RECIDIVISM ===')
    print('Two-year recidivists:', recid[1] , '(', percentages[1], '%)')
    print('Non-two-year recidivists:', recid[0], '(', percentages[0], '%)') 
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['Did not recidivize within 2 years', 'Recidivized within 2 years']
    plt.title("2-Year Recidivism Outcome")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('2-Year Recidivism Outcome')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(degree) / num_people) for degree in charge_degree] 
    print()
    print('=== CHARGE DEGREE ===')
    print('Misdemeanor:', charge_degree[0] , '(', percentages[0], '%)')
    print('Felony:', charge_degree[1], '(', percentages[1], '%)') 
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['Misdemeanor', 'Felony']
    plt.title("Charge Degree")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Charge Degree')
    plt.ylabel('Percentage')
    plt.show()

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
    percentages = [round(100 * (float)(violence) / num_recid) for violence in recid_violent] 
    print()
    print('=== VIOLENCE AMONG RECIDIVISTS ===')
    print('Total # or recidivists: ', num_recid)
    print('Recidivists who were violent:', recid_violent[1] , '(', percentages[1], '%)')
    print('Recidivists who were nonviolent:', recid_violent[0], '(', percentages[0], '%)')
    x_range = range(len(percentages))
    width = 1 / 2 
    x = ['Nonviolent', 'Violent']
    plt.title("Violence Among Recidivists")
    plt.bar(x, percentages, width, color="blue")
    plt.xlabel('Whether Offense Was Violent Among Recidivists')
    plt.ylabel('Percentage')
    plt.show()

def recid_prison_duration():
    DURATION_COL = 10
    OVERALL_RECID_COL = 11
    num_people = 7214
    duration = []
    recid_duration = np.zeros((2))
    num_recid = 0
    with open('../Data/Processed_compas-scores-two-years-reprocess.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[DURATION_COL] == '':
                continue
            num_recid += 1
            duration.append(int(row[DURATION_COL]))
            recid_duration[(int)(row[OVERALL_RECID_COL])] += int(row[DURATION_COL])
        duration_np = np.asarray(duration)
        mu = np.sum(duration) / num_recid 
        std_dev = np.std(duration)
    print()
    print('=== PRISON DURATION AMONG RECIDIVISTS ===')
    print('NOTE: Some outliers in duration skewed metrics (i.e. several people were imprisoned for 40 000+ days, whereas most were imprisoned for several hundred days')
    print('NOTE: Sentence duration data unavailable for non-recidivists')
    print('Mean: %.3f'%mu)
    print('Standard deviation: %.3f'%std_dev)
    outliers = [num for num in duration if num > 1000]
    outliers_np = np.asarray(outliers)
 
    f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
    ax.set_title('Prison duration among recidivists')
    ax.set_xlabel('Sentence duration (days)')
    ax.set_ylabel('Frequency (# of inmates)')

    ax.set_xlim(0, 1000)
    ax.hist(duration_np, bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ax2.set_xlim(40000, 50000)
    ax2.hist(duration_np)
    plt.ylim(0, 1200) 
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    plt.show()


print()
overview()
age()
race()
gender()
overall_recid()
two_year_recid()
charge_degree()
recid_violent()
recid_prison_duration()
print()
