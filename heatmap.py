
import numpy as np              # used to build the fake numbers below
import pandas as pd             # holds our data in a table (like an Excel sheet)
import matplotlib.pyplot as plt # draws the actual charts
import seaborn as sns           # makes matplotlib charts easier to build/prettier

file_name = "maxresults.csv"

#put csv into dataframe (df)
df = pd.read_csv(file_name)

#view first 5 rows
print(df.head())
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