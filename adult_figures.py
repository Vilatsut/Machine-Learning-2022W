# %%
import pandas as pd
# %%
df_adults = pd.read_csv(r'C:\Users\beregszaszim\Desktop\TU_Wien\ML\Exercise_0\adults\adults-clean.csv')
# %% education_num chart
df_adults_figure1 = df_adults[['education-num','class']]
df_adults_figure1_dummies = pd.get_dummies(df_adults_figure1, columns=['class'])
# %%
df_adults_figure1_grouped = df_adults_figure1_dummies.groupby(by='education-num').sum()
df_adults_figure1_grouped.rename(columns={'class_ <=50K':'<=50K', 'class_ >50K':'50K'}, inplace = True)
# %%
import matplotlib.pyplot as plt

df_adults_figure1_grouped.plot(kind='bar', stacked=True)
plt.savefig(r'C:\Users\beregszaszim\Desktop\TU_Wien\ML\Exercise_0\adults\adults_figure.png', dpi=200)
plt.show()
# %%
