import numpy as np              # used to build the fake numbers below
import pandas as pd             # holds our data in a table (like an Excel sheet)
from statsmodels.multivariate.manova import MANOVA  # runs the MANOVA math


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

# -----------------------------------------------------------------------
# HOW TO READ THIS (printed once per network, so you'll see this pattern twice):
#
#   viscosity                  -> does viscosity alone shift the combined
#                                 profile, regardless of diabetic status?
#                                 (expected to be significant -- that's the
#                                 basic dose-response relationship)
#
#   C(diabetic)                 -> *** direct healthy vs diabetic comparison ***
#                                 does diabetic status shift the combined
#                                 profile, averaged across viscosity levels?
#
#   viscosity:C(diabetic)       -> *** the more precise healthy vs diabetic
#                                 comparison *** -- does the effect of
#                                 viscosity on the combined profile change
#                                 depending on diabetic status? This is the
#                                 multivariate version of what your
#                                 regression's viscosity:C(diabetic) term
#                                 already tests one output at a time.
#
# For each row, look at "Pillai's trace" or "Wilks' lambda" and its
# associated Pr > F (the p-value). p < 0.05 is usually read as
# "statistically significant."
#
# Compare the two networks' results side by side: if the
# viscosity:C(diabetic) row is significant in BOTH networks with a similar
# pattern, that's strong evidence the diabetic effect is real and
# consistent. If it's significant in only one, that's worth investigating.
# -----------------------------------------------------------------------