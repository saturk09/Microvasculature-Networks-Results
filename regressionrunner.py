import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

# ---------------------------------------------------------------------------
# FAKE DATA (delete this whole block once you have a real CSV)
# ---------------------------------------------------------------------------
np.random.seed(42)

networks = ['A', 'B']
diabetic_statuses = [0, 1]                 # 0 = non-diabetic, 1 = diabetic
viscosities = np.linspace(3.0, 5.0, 6)     # 6 viscosity steps, edit to match your sweep

rows = []
for net in networks:
    for diabetic in diabetic_statuses:
        for visc in viscosities:
            # base relationships loosely mimicking fluid dynamics behavior:
            # velocity decreases as viscosity increases, faster if diabetic
            # pressure increases as viscosity increases, faster if diabetic
            # shear stress increases as viscosity increases, faster if diabetic
            diabetic_penalty = 1.4 if diabetic else 1.0
            network_shift = 1.15 if net == 'B' else 1.0

            velocity = (0.6 - 0.06 * visc * diabetic_penalty) * network_shift
            pressure = (60 + 8 * visc * diabetic_penalty) * network_shift
            shear_stress = (0.5 + 0.25 * visc * diabetic_penalty) * network_shift

            # small noise so the regression has something to fit, not exact lines
            velocity += np.random.normal(0, 0.01)
            pressure += np.random.normal(0, 1.0)
            shear_stress += np.random.normal(0, 0.03)

            rows.append({
                'network': net,
                'diabetic': diabetic,
                'viscosity': visc,
                'velocity': velocity,
                'pressure': pressure,
                'shear_stress': shear_stress,
            })

df = pd.DataFrame(rows)
# ---------------------------------------------------------------------------
# END FAKE DATA
# ---------------------------------------------------------------------------

#pandas file readings
print("Data preview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nRows per group (network x diabetic):")
print(df.groupby(['network', 'diabetic']).size())

#reading dose-response curves for viscosity vs pressure, velocity, and shear stress
dvs = ['velocity', 'pressure', 'shear_stress']
networks = sorted(df['network'].unique())

fig, axes = plt.subplots(len(networks), len(dvs), figsize=(15, 5 * len(networks)), sharex=True)
if len(networks) == 1:
    axes = axes.reshape(1, -1)

for i, net in enumerate(networks):
    sub_net = df[df['network'] == net]
    for j, dv in enumerate(dvs):
        sns.lineplot(data=sub_net, x='viscosity', y=dv, hue='diabetic',
                     marker='o', ax=axes[i, j], palette={0: 'tab:blue', 1: 'tab:red'})
        axes[i, j].set_title(f'Network {net}: {dv}')
        axes[i, j].legend(title='Diabetic', labels=['Non-Diabetic', 'Diabetic'])

plt.tight_layout()
plt.savefig("dose_response_curves.png", dpi=300)
plt.show()

#running the physical regressions
results = {}

for net in networks:
    sub = df[df['network'] == net]
    results[net] = {}
    print(f"\n{'=' * 60}\nNETWORK {net}\n{'=' * 60}")
    for dv in dvs:
        model = ols(f'{dv} ~ viscosity * C(diabetic)', data=sub).fit()
        results[net][dv] = model
        print(f"\n--- {dv} ---")
        print(model.summary())


# summary table, tells us interaction coefficient (good for correlations), and p-value for significance

summary_rows = []
for net in networks:
    for dv in dvs:
        model = results[net][dv]
        term = [t for t in model.params.index if 'viscosity:C(diabetic)' in t]
        if term:
            t = term[0]
            summary_rows.append({
                'network': net,
                'output': dv,
                'interaction_coefficient': round(model.params[t], 5),
                'p_value': round(model.pvalues[t], 5),
                'significant_at_alpha_of_0.05?': model.pvalues[t] < 0.05,
            })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{'=' * 60}\nviscosity x diabetic interaction\n{'=' * 60}")
print(summary_df.to_string(index=False))