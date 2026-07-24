
import numpy as np              # used to build the fake numbers below
import pandas as pd             # holds our data in a table (like an Excel sheet)
import matplotlib.pyplot as plt # draws the actual charts
import seaborn as sns           # makes matplotlib charts easier to build/prettier

# -----------------------------------------------------------------------
# STEP 1: BUILD FAKE DATA
# (Delete this whole block later and replace with:
#   df = pd.read_csv("your_real_file.csv")
#  once real COMSOL results exist. Your CSV just needs columns named
#  exactly: network, diabetic, viscosity, velocity, pressure, shear_stress)
# -----------------------------------------------------------------------

np.random.seed(42)  # locks the "randomness" so you get the same fake numbers every run

networks = ['1', '2']            # your two vessel network geometries
diabetic_statuses = [0, 1]       # 0 = non-diabetic, 1 = diabetic
viscosities = np.linspace(3.0, 5.0, 6)  # 6 evenly-spaced viscosity values from 3.0 to 5.0

rows = []  # empty list -- we'll fill it with one dictionary per simulated row

for net in networks:                      # loop over network A, then network B
    for diabetic in diabetic_statuses:     # loop over non-diabetic (0), then diabetic (1)
        for visc in viscosities:           # loop over each of the 6 viscosity values

            # these two lines just make diabetic/network B react more strongly
            # to viscosity -- purely to make the FAKE data look realistic
            diabetic_penalty = 1.4 if diabetic else 1.0
            network_shift = 1.15 if net == 'B' else 1.0

            # fake formulas: each output responds to viscosity in a
            # direction that mimics real fluid dynamics (you will replace
            # these three lines entirely once real data exists)
            velocity = (0.6 - 0.06 * visc * diabetic_penalty) * network_shift
            pressure = (60 + 8 * visc * diabetic_penalty) * network_shift
            shearStress = (0.5 + 0.25 * visc * diabetic_penalty) * network_shift

            # add a little random "noise" so the numbers aren't perfectly
            # smooth lines -- mimics real-world measurement variation
            velocity += np.random.normal(0, 0.01)
            pressure += np.random.normal(0, 1.0)
            shearStress += np.random.normal(0, 0.03)

            # save this one row of fake data as a dictionary, add it to our list
            rows.append({
                'network': net,
                'diabetic': diabetic,
                'viscosity': visc,
                'velocity': velocity,
                'pressure': pressure,
                'shearStress': shearStress,
            })

df = pd.DataFrame(rows)  #convert the list of dictionaries into a proper data table

print(df.head())  #sanity check -- print the first 5 rows

#creates the heatmap

corr_vars = ['viscosity', 'velocity', 'pressure', 'shearStress']  #columns to compare

corr_matrix = df[corr_vars].corr()  #calculates correlation between every pair of these 4 columns
#4x4 matrix filled with numbers betweej 0 and 1

plt.figure(figsize=(6, 5)) #start a set-sized figure

sns.heatmap(
    corr_matrix,
    annot=True,        #print the correlation inside each cell
    cmap='coolwarm',    #blue = neg, red = pos
    vmin=-1, vmax=1,     #set the color scale 
    center=0,            #0 = white
    fmt='.2f',            #round to hundredths
)
plt.title('Diabetic/Non-Diabetic Correlation Matrix')  #chart title
plt.tight_layout()                        #cleans up spacing
plt.savefig('correlation_overall.png')    #saves the chart as img
plt.show()                                #opens chart

#helpful splitting to prevent data accuracy via less generic correlations

fig, axes = plt.subplots(1, 2, figsize=(13, 5))   #1 row, 2 charts
group_labels = {0: 'Non-Diabetic', 1: 'Diabetic'}  #naming charts

for ax, status in zip(axes, [0, 1]): #non-diabetic before diabetic, iterates through each section
    subset = df[df['diabetic'] == status][corr_vars]  #keep only rows matching this group
    sns.heatmap(
        subset.corr(), #correlation matrix for subgroup
        annot=True, cmap='coolwarm',
        vmin=-1, vmax=1, center=0, fmt='.2f',
        ax=ax, #draw value into chart
    )
    ax.set_title(group_labels[status])#label chart

plt.tight_layout()
plt.savefig('correlation_by_group.png')
plt.show()