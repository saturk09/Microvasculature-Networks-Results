import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

networks = ['1', '2']
diabeticstatuses = [0, 1]

file = "ptresults.csv"

#put csv into dataframe (df)
df = pd.read_csv(file)

print(df.head())


#pandas file readings
print("Data preview:")
print(df.head())    #prints first 5 rows of table
print(f"\nShape: {df.shape}") #returns number of rows/columns to read
print(f"\nRows per group (network, diabetic)")
print(df.groupby(['network', 'diabetic']).size()) #splits data into groups for combinations of network/diabetes

#reading dose-response curves for viscosity vs pressure, velocity, and shear stress
dvs = ['velocity', 'pressure', 'shearStress'] #list of dependent variables
networks = sorted(df['network'].unique()) #grabs network album and returns no duplicates

fig, axes = plt.subplots(len(networks), len(dvs), figsize=(15, 5 * len(networks)), sharex=True) #just sets image sizing
if len(networks) == 1: #line recommended to add
    axes = axes.reshape(1, -1)

for i, net in enumerate(networks): #enumerate() gives position number and network value
    subnet = df[df['network'] == net] #filters dataframe to match network
    subnet = subnet.copy() #recommended to reduce errors
    subnet['diabetic'] = subnet['diabetic'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
    for j, dv in enumerate(dvs): #loop to give column position and actual name of dependent variables
        sns.lineplot(data=subnet, x='viscosity', y=dv, hue='diabetic', #palette keys
        marker='o', ax=axes[i, j], palette={'Non-Diabetic': 'tab:blue', 'Diabetic': 'tab:red'}) #draws line chart for variables
        axes[i, j].set_title(f'Network {net}: {dv}') #labels chart panel
        axes[i, j].legend(title='Diabetic Retinopathy') #adds legend box for blue + red definitions

plt.tight_layout() #adjusts spacing
plt.savefig("dose_response_curves.png", dpi=300) #saves whole chart grid as an image
plt.show() #opens a window on computer

#running the physical regressions
results = {} #stores every regression model

for net in networks: #loop through each network 
    sub = df[df['network'] == net] 
    results[net] = {}
    print(f"\nNETWORK {net}") #prints network name
    for dv in dvs:
        model = ols(f'{dv} ~ viscosity * C(diabetic)', data=sub).fit() #ols = ordinary least squares, linear regression model to explain output variable
        results[net][dv] = model #saves fitted model into results
        print(f"\n--- {dv} ---")
        print(model.summary())  #prints a small header with statistical results


#summary table for significance/visuals

summaryrows = [] #empty list to hold 1 dictionary/network
for net in networks: #loops through every network/output variable
    for dv in dvs:
        model = results[net][dv]
        term = [t for t in model.params.index if 'viscosity:C(diabetic)' in t] #lists all variable names in fitted model
        if term:
            t = term[0]
            summaryrows.append({ #adding the matching names to folder
                'network': net,
                'output': dv,
                'interaction_coefficient': round(model.params[t], 5),
                'p_value': round(model.pvalues[t], 5),
                'significant_at_alpha_of_0.05?': model.pvalues[t] < 0.05,
            })

summarydf = pd.DataFrame(summaryrows) #converts dictionaries to a table
print(f"\n\nviscosity x diabetic interaction\n")
print(summarydf.to_string(index=False)) #prints a header then summary table