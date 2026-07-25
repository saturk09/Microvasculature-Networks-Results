import numpy as np              #used to build the fake numbers below
import pandas as pd             #holds data in a table
from statsmodels.multivariate.manova import MANOVA  #runs the MANOVA math


networks = ['1', '2']
diabetic_statuses = [0, 1]

file_name = "ptresults.csv"

#put csv into dataframe (df)
df = pd.read_csv(file_name)
networks = sorted(df['network'].unique()) #taking the networks, sorted

#view first 5 rows
print(df.head())

results = {}  #stores results

for net in networks:                              #loop through network 1, then network 2
    sub = df[df['network'] == net]                 #keep only network's rows

    print(f"NETWORK {net}: Healthy vs Diabetic, across viscosity")

    mancova_model = MANOVA.from_formula(
        'pressure + shearStress ~ viscosity * C(diabetic)',
        data=sub,
    )
    mancova_results = mancova_model.mv_test()      #runs test
    results[net] = mancova_results                  #saves for later

    print(mancova_results)