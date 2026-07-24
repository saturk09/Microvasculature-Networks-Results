#note that this is just a supporting test. MANOVA does not take continuous data points, only discrete given ones.

import numpy as np              # used to build the fake numbers below
import pandas as pd             # holds our data in a table (like an Excel sheet)
from statsmodels.multivariate.manova import MANOVA  # runs the MANOVA math

# -----------------------------------------------------------------------
# STEP 1: BUILD FAKE DATA
# (Same fake-data block as the other two scripts, so all three stay
# consistent. Delete this later and replace with:
#   df = pd.read_csv("your_real_file.csv")
#  once real COMSOL results exist.)
# -----------------------------------------------------------------------

np.random.seed(42)  # locks the "randomness" so you get the same fake numbers every run

networks = ['1', '2']            #metworks
diabetic_statuses = [0, 1]       #0 = non-diabetic, 1 = diabetic
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
            shear_stress = (0.5 + 0.25 * visc * diabetic_penalty) * network_shift

            # add a little random "noise" so the numbers aren't perfectly
            # smooth lines -- mimics real-world measurement variation
            velocity += np.random.normal(0, 0.01)
            pressure += np.random.normal(0, 1.0)
            shear_stress += np.random.normal(0, 0.03)

            # save this one row of fake data as a dictionary, add it to our list
            rows.append({
                'network': net,
                'diabetic': diabetic,
                'viscosity': visc,
                'velocity': velocity,
                'pressure': pressure,
                'shear_stress': shear_stress,
            })

df = pd.DataFrame(rows)  # convert the list of dictionaries into a proper data table

print(df.head())  # sanity check -- print the first 5 rows

#picking low, medium, and high viscosity values after the sweep

all_viscosities = sorted(df['viscosity'].unique())  # very distinct viscosity value present with the pandas sorted commands
print(f"\nAvailable viscosity values: {all_viscosities}")

#grabbing the lowest, middle, and highest values from sweeps
fixed_viscosities = [
    all_viscosities[0],                            #lowest value in the sweep
    all_viscosities[len(all_viscosities) // 2],     #roughly the middle value
    all_viscosities[-1],                            #highest value in the sweep
]
print(f"Fixed viscosities: {fixed_viscosities}")

#filtering data to just those values

manova_data = df[df['viscosity'].isin(fixed_viscosities)]  #keeping matching rows

print(f"\nRows being used for MANOVA: {len(manova_data)}")
print(manova_data.groupby(['network', 'diabetic']).size())  #look at group sizes

#manova running

manova_model = MANOVA.from_formula(
    'velocity + pressure + shear_stress ~ C(network) * C(diabetic)',
    data=manova_data,
)

manova_results = manova_model.mv_test()  #uses method to run test
print("results")
print(manova_results)

# -----------------------------------------------------------------------
# HOW TO READ THIS (given your research question is really about DIABETES,
# and network is just two different anatomical layouts you're checking
# the diabetic effect against):
#
#   C(network)                -> network A vs B differ in baseline flow.
#                                 EXPECTED and not a problem for your story --
#                                 two different vessel geometries SHOULD have
#                                 different baseline numbers. Don't worry if
#                                 this row is significant.
#
#   C(diabetic)                -> does diabetic status alone shift the
#                                 combined profile, averaged across both
#                                 networks? Relevant, but not the main event.
#
#   C(network):C(diabetic)     -> *** THIS IS THE ROW THAT MATTERS MOST ***
#                                 Does the diabetic effect DEPEND on which
#                                 network you're looking at?
#                                   - NOT significant here = good news:
#                                     the diabetic effect is consistent
#                                     across both networks (it replicates).
#                                   - Significant here = the diabetic effect
#                                     shows up differently depending on
#                                     network, which is worth digging into.
#
# For each row, look at "Pillai's trace" or "Wilks' lambda" and its
# associated Pr > F (the p-value). p < 0.05 is usually read as
# "statistically significant."
# -----------------------------------------------------------------------