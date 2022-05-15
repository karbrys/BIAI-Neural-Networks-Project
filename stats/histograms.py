import seaborn as sns
import matplotlib.pyplot as plt

def create_survival_age_histogram(train_df):
    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)
    plt.savefig('hist_age_survival.png')
    plt.show()
    return g

def create_survival_pclass_histogram(train_df):
    g = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    g.map(plt.hist, 'Age', alpha=.5, bins=20)
    g.add_legend()
    plt.savefig('hist_pclass_survival.png')
    plt.show()
    return g

def create_correlation(train_df):
    corr_train = train_df.corr()
    sns.heatmap(corr_train)
    plt.savefig('correlation.png')
    plt.show()
    return corr_train