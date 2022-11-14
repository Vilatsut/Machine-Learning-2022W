# %%
import pandas as pd
import os
# %%
df_car_evaluation = pd.read_csv(os.getcwd() + '/data/car.data', sep=",")
# %%
df_car_evaluation.describe()
# %% Piechart about the classes of the car_evaluation dataset
import matplotlib.pyplot as plt

labels = df_car_evaluation['class'].unique()
sizes = [70.023, 22.222, 3.762, 3.993]
explode = (0.1, 0, 0, 0) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=200)
ax1.axis('equal') 

plt.savefig('figures/car_evaluation_classes.png', dpi=200)
plt.show()
# %%
