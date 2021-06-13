#####################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
#####################################################

# 1. Data Preparation
# 2. Expected Sales Forecasting with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
# 5. Creating Segments by CLTV
# 7. Submitting Results to Database

#####################################################
# 1. Data Preparation
#####################################################

################################
# Required Library and Functions
################################

!pip install mysql-connector-python-rf
!pip install mysql
!pip install lifetimes
!pip install sqlalchemy
from sqlalchemy import create_engine # library that allows us to add and delete data in sql tables
import datetime as dt
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# credentials.

creds = {'user': '...',
         'passwd': '...',
         'host': '...',
         'port': ...,
         'db': '...'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'
conn = create_engine(connstr.format(**creds))
# conn.close()

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)
pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)

df = retail_mysql_df.copy()

# 75% -> 10, max -> 80k => problem! => outliers should be discarded
df.describe().T

# missing values are discarded
df.dropna(inplace = True)

df = df[df["Quantity"] > 0]

# The 'C' in the invoice shows the canceled transactions. Canceled transactions leave the dataset.
df = df[~df["Invoice"].str.contains("C", na = False)]
df.describe().T

# suppressed with outlier threshold
# suppressed by calculated quarter values
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)
df.head()

# Selecting UK customers
df = df[df["Country"] =="United Kingdom"]


################################
# # Preparation of Lifetime Data Structure
################################

# recency: The elapsed time since the last purchase. Weekly. (according to analysis day on cltv_df, user specific here)
# T: The age of the customer. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1) -> Theoretically, must be greater than 1
# monetary_value: average earnings per purchase

# frequency>1 => churn became a client, not a client.

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days, # -> recency
                                                        lambda date:(today_date-date.min()).days], # -> tenure
                                        'Invoice': lambda num: num.nunique(), # -> frequency
                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()}) # -> monetary
# When it is run, an index problem occurs. Lambdas are hierarchical.

# First-order nomenclatures are discarded.
cltv_df.columns = cltv_df.columns.droplevel(0)

# New column names are assigned
cltv_df.columns = ["recency","T","frequency","monetary"]

# Monetary value should be expressed as average earnings per purchase.
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]


cltv_df = cltv_df[cltv_df["monetary"] > 0]

# Expression of recency and T for BGNBD in weekly terms
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency must be greater than 1
# If the frequency is equal to one, this person is not a customer.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]


############################
# Establishing the BG-NBD Model (Probabilistic acquisitions models)
############################

bgf = BetaGeoFitter(penalizer_coef=0.001)
# Penalties have been applied to the weights to prevent overfitting.

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
# a, beta, gamma values, found.

# Only purchases are considered.
# 1 week expected transaction
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
# 1 month expected transaction
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


#######################################
# Establishment of GAMMA-GAMMA Model (Customer's profit per transaction)
########################################

# add penalty coefficient to the function
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

# returns customer specific values
ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary']).head(10)

# contingent expected average profits of customers
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

# customers' conditional expected average profits are added to the data set.
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.head()
cltv_df.sort_values("expected_average_profit", ascending=False).head(20)


#######################################
# Calculation of CLTV with BG-NBD and GG model.
#######################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 month
                                   freq="W", # T's frequency information
                                   discount_rate=0.01)
cltv.head()

cltv = cltv.reset_index()  # id de-indexed
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left") # Merged according to cltv_df
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Standardization of CLTV
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head()


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 month
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index() # id de-indexed
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  #12 month
                                   freq="W",
                                   discount_rate=0.01)
cltv = cltv.reset_index() # id index olmaktan çıkarıldı
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Comment: Although there were differences in CLV values when the first 10 observations were examined, there was no change in other criteria.
# No meaningful relationship was found between them.
# Although the receny and t values of the person in the first row are lower than the other people, they have the highest clv value.
# This may be due to the fact that the monetary value has suppressed other parameters.

plot_period_transactions(bgf)
plt.show()

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 month
                                   freq="W",
                                   discount_rate=0.01)
cltv = cltv.reset_index() # id index olmaktan çıkarıldı
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")


#################################################
# Creating Segments According to CLTV
################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"]) # qcut = çeyrek değerlere göre gruplama yaptırır.
cltv_final.head()
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# A segment's frequency, that is, the frequency of shopping and the monetary (purchasing) frequency, is ahead of other segments. Priority should be given to the works to be carried out specifically for the customers.
# Secondly, based on frequency and monetary values, priority can be given to B segment campaigns in order to position them as potential customers.

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels = ["D","C","B","A"])

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)

cltv_final.to_sql(name = 'mihricankizilyer', con=conn, if_exists = 'replace', index = False)



