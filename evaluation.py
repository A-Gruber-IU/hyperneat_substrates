import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# load CSV file
df = pd.read_csv("results.csv")

# set variables for categories
df['method'] = df['method'].astype('category')
df['sampling'] = df['sampling'].astype('category')
df['dimensionality'] = df['dimensionality'].astype('category')

# ANOVA model with interaction between factors
# model = smf.ols('fitness ~ C(method) * C(sampling) * C(dimensionality)', data=df).fit()

# ANOVA model without interaction between factors
model = smf.ols('fitness ~ C(reduction) + C(sampling) + C(dimensionality)', data=df).fit()

# ANOVA table (Typ 2)
anova_table = sm.stats.anova_lm(model, typ=2)

# Show results
print("ANOVA Results:")
print(anova_table)
