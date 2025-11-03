import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from scipy.stats import anderson,normaltest,zscore,iqr,chisquare,pearsonr,spearmanr,pointbiserialr,levene,chi2_contingency
from statsmodels.api import qqplot
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from feature_engine.encoding import RareLabelEncoder,OneHotEncoder
from category_encoders import TargetEncoder
import pingouin as pg
from itertools import combinations
from scipy.stats import mannwhitneyu
import sys
import umap.umap_ as umap
from sklearn.manifold import TSNE
sys.path.append("..\src")
from PY_Class_Def import RobustZ_Score,check_pca_separation
import warnings

#loading the dataset
df=pd.read_csv(r'..\DataSources\Processed\Cleaned_Final_Stats_test.csv').iloc[:,1:]
print('Dataset loaded')
print('\n')
print('\n')

print('-----------------------------Number of NaN Value-----------------------------')
print('\n')
null_feat=[]
#with the mean we get "percentage values"
df_null=df.isnull().mean().to_frame().transpose()
for x in list(df_null.columns):
    if df_null[x][0] > 0.0:
        null_feat.append(x)
print(f'Number of of columns with Nan values: {len(null_feat)}')
print(f'Columns with Nan values: {null_feat}')
print('\n')
print('\n')

print('-----------------------------Splitting into train & test-----------------------------')
#dropping 
X=df.drop(['League_Joined','Club_Joined','Transfer_Fee'],axis=1)
y=df['League_Joined']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
print('Dataset splitted in train & test sets')
print('\n')
print('\n')


print('##################################NUMERICA FEATURES#########################################')
print('\n')
print('-----------------------------Amount of Numeric and Categorical Features-----------------------------')
X_train_num=X_train.select_dtypes(exclude='object')
X_train_cat=X_train.select_dtypes(include='object')
print('\n')
print(f'Number of only Numeric Features: {X_train_num.shape[1]}')
print(f'Number of only Categorical Features: {X_train_cat.shape[1]}')
print('\n')
print('\n')

print('-----------------------------Checking Distribution-----------------------------')
print("D'Angostin Distribution Test")
print('\n')
normal=[]
non_normal=[]

for x in list(X_train_num.columns):
    stat,p_value=normaltest(X_train_num[x])
    if p_value > 0.05:
        normal.append(x)
    else:
        non_normal.append(x)

print(f'Normal Distribution: {normal}')
print(f'Non Normal Distribution: {non_normal}')

print('\n')
print("Anderson Distribution Test")
print('\n')
normal=[]
non_normal=[]

for x in list(X_train_num.columns):
    result=anderson(X_train_num[x],dist='norm')
    #determine the significane level
    index=result.significance_level.tolist().index(5.0)
    cv=result.critical_values[index]
    if result.statistic > cv:
        non_normal.append(x)
    else:
        normal.append(x)

print(f'Normal Distribution: {normal}')
print(f'Non Normal Distribution: {non_normal}')

print('\n')
print('\n')

print('-----------------------------Checking Distribution-----------------------------')
print('Outlier detection IQR')
outlier_list=[]
for x in list(X_train_num.columns):
    q1=X_train_num[x].quantile(0.25)
    q3=X_train_num[x].quantile(0.75)
    iqr=q3-q1

    lower_bound=q1-(iqr*1.5)
    upper_bound=q3+(iqr*1.5)

    outliers=len(X_train_num[(X_train_num[x]<lower_bound) | (X_train_num[x]>upper_bound)])
    if outliers > 0:
        v=f'{x}-{(round(outliers/len(X_train_num),2))*100}'
        outlier_list.append(v)
print((len(outlier_list)/len(list(X_train_num.columns)))*100)
print(outlier_list)
print('\n')
print('Outlier detection Robust Z-Score')
outlier_z=[]
for x in list(X_train_num.columns):
    rbzs=abs(RobustZ_Score(X_train_num[x]))
    if np.any(rbzs > 3):
        outlier_z.append(f'{x}-{(round(sum(rbzs > 3)/len(X_train_num),2))*100}%')
print((len(outlier_z)/len(list(X_train_num.columns)))*100)
print(outlier_z)

print('\n')
print('\n')

print('Boxplots, QQ-Plots & Histograms')
print('Look it up in the notebook "Exploratory_Transfer_Target_Model".')

print('\n')
print('\n')

print('-----------------------------Corrleation & Multicollinarity-----------------------------')
print('Correlation')
#here we get a list of features which could be filter out, because they are correlated too strong with other X features and have a weaker variance
##by using a set, we make sure not to add duplicate values to teh list
corr_matrix=X_train_num.corr(method='pearson').abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop=set()
for c in list(upper.columns):
    for x in list(upper.index):
        if upper.loc[x,c] >= 0.6:
            if X_train_num[c].var() > X_train_num[x].var():
                to_drop.add(x)
            else:
                to_drop.add(c)

print(f"Number of affected Columns that causes Multicollinarity: {len(to_drop)}")
print(f"Percentage of affected Features: {round((len(to_drop)/len(list(X_train_num.columns)))*100,2)}")
print('\n')
print('VIF')
#drop or replace inf and NaNs
X_train_num_clean=X_train_num.replace([np.inf,-np.inf],np.nan).dropna()

#add constant for intercept
x_vif=sm.add_constant(X_train_num_clean)

#compute VIF for each feature
vif_df=pd.DataFrame()
vif_df['Features']=X_train_num_clean.columns
vif_df['VIF']=[variance_inflation_factor(x_vif.values,i) for i in range(1,x_vif.shape[1])]

#handle 'inf' and high VIFs cleanly
vif_df['VIF']=pd.to_numeric(vif_df['VIF'],errors='coerce')
high_vif_df=vif_df[vif_df['VIF']>10].sort_values('VIF',ascending=False)
inf_vif_df=vif_df[vif_df['VIF']==float('inf')]

VIF=pd.concat([high_vif_df,inf_vif_df]).drop_duplicates()
#Note => we do not need to check for multicollinarity
print(f"Number of affected Columns that causes Multicollinarity: {len(VIF)}")
print(f"Percentage of affected Features: {round((len(VIF)/len(X_train_num.columns))*100,2)}")
print(VIF)


print('\n')
print('\n')
print('\n')


print('##################################NUMERICA FEATURES#########################################')
print('y Class appreances')
((y_train.value_counts()/len(y_train))*100).to_frame().reset_index().rename(columns={'count':'Percentage of Appearance'})
print('\n')
print('\n')


rare_nation=RareLabelEncoder(tol=0.005,variables='Nation')
adjusted_nation_X_train_cat=rare_nation.fit_transform(X_train_cat)

cl_rare=RareLabelEncoder(tol=0.0025,variables=['Club_Left'])
adjusted_cl_X_train_cat=cl_rare.fit_transform(adjusted_nation_X_train_cat)

ll_rare=RareLabelEncoder(tol=0.002,variables=['League_Left'])
adjusted_ll_X_train_cat=ll_rare.fit_transform(adjusted_cl_X_train_cat)

ohe_X=OneHotEncoder(drop_last=False,variables=['Pos','Age'])
adjusted_ohe_X_train_cat=ohe_X.fit_transform(adjusted_ll_X_train_cat)

tar_code=TargetEncoder(cols=['Nation','Club_Left','League_Left'])
tar_code.fit(adjusted_ohe_X_train_cat,y_train)
adjusted_mean_X_train_cat=tar_code.transform(adjusted_ohe_X_train_cat)

df_median_encoded = adjusted_mean_X_train_cat.copy()
df_median_encoded['y_train'] = y_train


print('Linearity Check for Encoded Features')
print('Look it up in the notebook "Exploratory_Transfer_Target_Model".')
print('\n')


print('25th, median, 75th of encoded features')
for ec in ['Nation','Club_Left','League_Left']:
    print(f'{ec}')
    print(df_median_encoded.groupby('y_train')[ec].agg(
        q25=lambda x: x.quantile(0.25),
        median='median',
        q75=lambda x: x.quantile(0.75)
    ))
    print('\n')
    print('\n')

print('\n')
print('Point Biserial Correlation')
ohe_y = OneHotEncoder(drop_last=False, variables=['y_train'])
corr_ohe_df = ohe_y.fit_transform(df_median_encoded)

point_corr=[]
for ec in ['Nation','Club_Left','League_Left']:
    for ohe in ['y_train_Ligue 1','y_train_LaLiga', 'y_train_Premier League', 'y_train_Serie A','y_train_Bundesliga', 'y_train_Other Leagues']:
        stats,p_value=pointbiserialr(corr_ohe_df[ec],corr_ohe_df[ohe])

        point_corr.append(
            {
                "Combinations":f"{ec}-{ohe}",
                "Correlation Level":stats,
                "P-Value":p_value
            }
        )
point_corr_df=pd.DataFrame(point_corr)
print(point_corr_df[point_corr_df['P-Value']<0.05])
print('\n')
print('Linearity CHeck of all X features based on y classes')
step_df=pd.concat([X_train_num,adjusted_mean_X_train_cat],axis=1)
final_adjust_df=pd.concat([step_df,y_train],axis=1)
x_feat=list(final_adjust_df.columns)
for d in ['Player','Squad','League','Pos_MF','Pos_FW','Pos_GK','Pos_DF','Age_Early Twenties','Age_Late Twenties','Age_Teenager','Age_Early Thirties','Age_Late Thirties']:
    x_feat.remove(d)

for x in x_feat:
    if x == 'League_Joined':
        continue
    else:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=final_adjust_df,y=x,hue='League_Joined')
        plt.legend(bbox_to_anchor=[1.001, 1.001])
        plt.title(f'Linearity Check for y_train and {x}',size=16,fontweight='bold')
        print(f'{x}')
        print(final_adjust_df.groupby(['League_Joined'])[x].agg(
            q25=lambda x: x.quantile(0.25),
            variance="var",
            mean='mean',
            median='median',
            q75=lambda x: x.quantile(0.75)
        ))
        print('\n')
        print('\n')
print('The graphics are avaiable on "Exploratory_Transfer_Target_Model"')
print('\n')
print('\n')

print('################################## ANOVA Test #########################################')
print('Equal Variance Assumption')
x_feat=list(final_adjust_df.columns)
for d in ['Player','Squad','League','Pos_MF','Pos_FW','Pos_GK','Pos_DF','Age_Early Twenties','Age_Late Twenties','Age_Teenager','Age_Early Thirties','Age_Late Thirties','League_Joined']:
    x_feat.remove(d)

equal_variance=[]
unequal_variance=[]
for x in x_feat:
    evt=pg.homoscedasticity(data=final_adjust_df, dv=x, group='League_Joined', method='levene')
    if evt['equal_var'].values[0]:
        equal_variance.append(x)
    else:
        unequal_variance.append(x)
print(f'Equal Variance: {equal_variance}')
print(f'Apperances of Equal Variance: {len(equal_variance)}')
print(f'Unequal Variance: {unequal_variance}')
print(f'Apperances of unequal Variance: {len(unequal_variance)}')
print('\n')

print('Normality Assumption')
x_feat=list(final_adjust_df.columns)
for d in ['Player','Squad','League','Pos_MF','Pos_FW','Pos_GK','Pos_DF','Age_Early Twenties','Age_Late Twenties','Age_Teenager','Age_Early Thirties','Age_Late Thirties','League_Joined']:
    x_feat.remove(d)

normal=[]
non_normal=[]

for c in list(final_adjust_df['League_Joined'].unique()):
    for x in x_feat:
        input_value=final_adjust_df[final_adjust_df['League_Joined']==c][x]
        stats,p_value=normaltest(input_value)
        if p_value < 0.05:
            non_normal.append(f'{c}-{x}')
        else:
            normal.append(f'{c}-{x}')
print(f'Normal Distrbution: {normal}')
print(f'Appearances of Normal Distrbution: {len(normal)}')
print(f'Non-Normal Distrbution: {non_normal}')
print(f'Appearances of Non-Normal Distrbution: {len(non_normal)}')

print('\n')
print('Kruskal-Wallis (ANOVA) & Dunn Test (Ado Hoc/Pairwise Test)')
# 1) Kruskal across all numeric features
drop_cols = {'Player','Squad','League','Pos_MF','Pos_FW','Pos_GK','Pos_DF',
             'Age_Early Twenties','Age_Late Twenties','Age_Teenager',
             'Age_Early Thirties','Age_Late Thirties','League_Joined','MP'}
x_feat = [c for c in final_adjust_df.columns if c not in drop_cols]

kw = pg.kruskal(data=final_adjust_df, dv='MP', between='League_Joined')
kw.index = ['MP']

for x in x_feat:
    kw_loop = pg.kruskal(data=final_adjust_df, dv=x, between='League_Joined')
    kw_loop.index = [x]
    kw = pd.concat([kw, kw_loop], axis=0)

# Significant features after Kruskal (uncorrected p; optionally FDR-correct)
ssf = kw.index[kw['p-unc'] < 0.05].tolist()

# 2) Pairwise Mann–Whitney + signed rank-biserial effect sizes
all_pairwise = []

for s in ssf:
    df_pw = final_adjust_df[[s, 'League_Joined']].copy()
    df_pw[s] = pd.to_numeric(df_pw[s], errors='coerce')
    df_pw = df_pw.dropna(subset=[s, 'League_Joined'])

    # Pairwise nonparam tests with Holm correction
    dt = pg.pairwise_tests(dv=s, between='League_Joined',
                           data=df_pw, parametric=False, padjust='holm')

    # Compute signed rank-biserial via AUC per pair
    pairs = []
    leagues = sorted(df_pw['League_Joined'].unique())

    for a, b in combinations(leagues, 2):
        x = df_pw.loc[df_pw['League_Joined'] == a, s].values
        y = df_pw.loc[df_pw['League_Joined'] == b, s].values
        n1, n2 = len(x), len(y)
        if n1 >= 2 and n2 >= 2:
            # Directional U: probability(X > Y)
            U_greater = mannwhitneyu(x, y, alternative='greater').statistic
            auc = U_greater / (n1 * n2)      # P(X>Y)
            r_rb = 2 * auc - 1               # signed rank-biserial
            pairs.append({'A': a, 'B': b, 'r_rb': r_rb})

    eff = pd.DataFrame(pairs)
    dt = dt.merge(eff, on=['A', 'B'], how='left')
    dt.insert(0, 'feature', s)  # keep track of which feature this table is for
    all_pairwise.append(dt)

pairwise_results = pd.concat(all_pairwise, ignore_index=True)
print(f'Number of unique statistically significant relationshipt between Features and Leagues Joined: {len(pairwise_results["feature"].unique())}')
print(pairwise_results["feature"].unique())
print(pairwise_results[pairwise_results['p-unc']<0.05])
print('\n')
print('\n')

print('################################## Chi-Square Test #########################################')
#using the dataframe with 
df_chi = adjusted_ll_X_train_cat.copy()
df_chi['y_train'] = y_train

for col in list(df_chi.columns)[:-1]:
# contingency table
    tbl = pd.crosstab(df_chi[col], df_chi['y_train'])

    # Pearson chi-square (asymptotic) + expected counts check
    chi2, p_asymp, dof, expected = chi2_contingency(tbl, correction=False)

    # inspect expected counts
    expected = pd.DataFrame(expected, index=tbl.index, columns=tbl.columns)
    prop_small = (expected < 5).mean().mean()  # proportion of cells < 5

    # permutation-based p-value (Monte Carlo)
    rng = np.random.default_rng(101)
    n_perm = 10000

    # observed stat
    obs_stat = chi2_contingency(tbl, correction=False)[0]

    # permute y labels to break association
    y = df_chi['y_train'].to_numpy()
    stats_perm = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        tperm = pd.crosstab(df_chi[col], y_perm)
        stats_perm[i] = chi2_contingency(tperm, correction=False)[0]

    p_perm = (1 + (stats_perm >= obs_stat).sum()) / (n_perm + 1)

    # Cramér's V (bias-corrected)
    n = tbl.values.sum()
    r, c = tbl.shape
    phi2 = obs_stat / n
    phi2corr = max(0, phi2 - (r-1)*(c-1)/(n-1))
    rcorr = r - (r-1)**2/(n-1)
    ccorr = c - (c-1)**2/(n-1)
    cramers_v = np.sqrt(phi2corr / max(1e-12, min(rcorr-1, ccorr-1)))

    print(f"{col} | Asymptotic p: {p_asymp:.4g} | Permutation p: {p_perm:.4g} | Cramér's V: {cramers_v:.3f} | % expected<5: {prop_small:.2%}")

print('\n')
print('\n')
print('################################## Dimension Reduction #########################################')
print('Look it up in the notebook "Exploratory_Transfer_Target_Model".')