import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None

# ---------------------------read in data and using data---------------------------
df = pd.read_csv(r'Users/caiguohua/Desktop/Movie Portfolio Project/movies.csv')
df.head()
# ---------------------------looking for missing data-------------------------------
for col in df.columns:
    pct_missing = np.mean(df[col].isnull)
    print('{} - {}%'.format(col, pct_missing))
# ---------------------------data cleaning----------------------------------------
# data type our columns
df.dtypes()
# change data type of columns
df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')
# create correct year column
df['year correct'] = df['released'].astype(str).str[:4]
# sort by gross desc
df.sort_values(by=['gross'], inplace=False, ascending=False)
# display max columns
df.set_option('display.max_rows',None)
# drop any duplicates
df.drop_duplicates()
# --------------------------find correlation in the data------------------------------
# budget high correlation
# company high correlation
# scatter plot with budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget VS Gross Earnings')
plt.xlabel('budget')
plt.ylabel('gross')
plt.show()
# plot budget vs  gross using seaborn
sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color': 'red'}, line_kws={'color': 'blue'})
plt.title('Budget VS Gross Earnings')
plt.xlabel('budget')
plt.ylabel('gross')
plt.show()
# looking at the correlation (pearson, kendall, spearman)
# numeric all the data of columns
numeric_df = df
for col_name in numeric_df:
    if numeric_df[col_name] == 'object':
        numeric_df[col_name].astype('category')
        numeric_df[col_name] = numeric_df[col_name].cat.codes
# plot the heatmap
correlation_mat = numeric_df.corr(method='pearson')
sns.heatmap(correlation_mat, annot=False)
plt.title("Correlation matrix for Numeric Features")
plt.xlabel('Movie features')
plt.ylabel('Movie features')
plt.show()
# check paris
corr_pairs = correlation_mat.unstack()
print(corr_pairs)
# sort paris
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
print(sorted_pairs)
# check strong paris
strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]
print(strong_pairs)

