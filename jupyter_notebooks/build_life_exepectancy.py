import pandas as pd
import numpy as np

# Set up person details DataFrame
cols = ['weight', 'smoker','physical_activity_scale', 'BMI', 'height', 'male']
person_details = pd.DataFrame(columns = cols)

# Set up people details

population = 1000

weight = np.random.randint(50 ,110, population)
person_details['weight'] = weight

smoking = np.random.random(population)
smoking = smoking >0.7
smoking = smoking.astype(int)
person_details['smoker'] = smoking

physical_actvity = np.random.randint(1, 11, population)
person_details['physical_activity_scale'] = physical_actvity

bmi = np.random.normal(27, 3, population)
bmi = bmi.astype(int)
person_details['BMI'] = bmi

height = ((weight / bmi) ** 0.5) * 100
height = height.astype(int)
person_details['height'] = height

male =np.random.randint(0, 2, population)
person_details['male'] = male

## Calculate life expectancy

def estimate_life(weight, smoker, activity, bmi, height, male):
    life = 90 # base life expetcancy
    
    # substract 10 years for smoking
    life -= smoker * 10
    
    # substract 5 years for being male
    life -= male * 5
    
    # Add 1 year for each activity level
    life += activity
    
    # reduce life expectancy for BMI 
    life -= (bmi ** 2) / 30
    
    # subtract random years
    life -= np.random.randint(0,10)
    
    life = int(life + 0.499)
    
    return life
    
life_exp = []
for index, row in person_details.iterrows():
    x = list(row)
    life_exp.append(estimate_life(*x))

person_details['life_expectancy'] = life_exp

person_details.to_csv('life_expectancy.csv', index=False)
