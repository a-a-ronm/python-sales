import pymysql
import pandas as pd
import statsmodels.api as sm

#Define variables for username, password, and database name in SQL server
us = ""
pw = ""
db = ""


#Connect to database in MySQL
connection = pymysql.connect(host='localhost',
                             user=us,
                             password=pw,
                             database=db,
                             charset='utf8mb4')

#Import product category data as data frame
with connection.cursor(pymysql.cursors.DictCursor) as cursor:
    cursor.execute('''SELECT O.GrossProfit,O.UnitsOrders,P.ProductCategory FROM ordersfact O 
                   JOIN products P ON O.ProductID = P.ProductID''')
    categories = cursor.fetchall()
df_cats = pd.DataFrame(categories)

#Add set of dummies for product categories
cat_dummy = pd.get_dummies(df_cats.ProductCategory, prefix = "Category")
df_cats = pd.concat([df_cats, cat_dummy], axis = 1)

df_cats.head()

#Define category dummies as regressors
X = df_cats.drop(["GrossProfit", "UnitsOrders", "ProductCategory"], axis=1)

#Regression model for profit by category
cat_profit = sm.OLS(df_cats[["GrossProfit"]], X).fit()
cat_profit.summary()

#Regression model for orders by category
cat_orders = sm.OLS(df_cats[["UnitsOrders"]], X).fit()
cat_orders.summary()

#Import order method data as data frame
with connection.cursor(pymysql.cursors.DictCursor) as cursor:
    cursor.execute('''SELECT O.GrossProfit,O.UnitsOrders,M.SalesOrderMethod FROM ordersfact O 
                   JOIN ordermethod M ON O.OrderMethodID = M.OrderMethodID''')
    methods = cursor.fetchall()    
df_methods = pd.DataFrame(methods)

#Encode order method as set of dummy variables and add to data frame
method_dummy = pd.get_dummies(df_methods.SalesOrderMethod, prefix = "Method")
df_methods = pd.concat([df_methods, method_dummy], axis = 1)

df_methods.head()

#Define method dummies as regressors
X = df_methods[["Method_Direct", "Method_Online", "Method_Retail"]]

#Model profit by method
method_profit = sm.OLS(df_methods[["GrossProfit"]], X).fit()
method_profit.summary()

#Model orders by method
method_orders = sm.OLS(df_methods[["UnitsOrders"]], X).fit()
method_orders.summary()

#Import customer tier data as data frame
with connection.cursor(pymysql.cursors.DictCursor) as cursor:
    cursor.execute('''SELECT O.GrossProfit,O.UnitsOrders,C.CustomerTier FROM ordersfact O 
                   JOIN customers C ON O.CustomerID = C.CustomerID''')
    tiers = cursor.fetchall()
df_tiers = pd.DataFrame(tiers)    
    
#Encode tiers as ordered categories for regression
#Bronze = 0, Silver = 1, etc
df_tiers["CustomerTier"] = df_tiers["CustomerTier"].astype("category")
df_tiers["CustomerTier"] = df_tiers["CustomerTier"].cat.set_categories(["Bronze", 
                                                                        "Silver", 
                                                                        "Gold", 
                                                                        "Platinum", 
                                                                        "PlatinumElite"], ordered=True)
df_tiers.sort_values("CustomerTier", inplace=True)
df_tiers["CustomerTier"] = df_tiers["CustomerTier"].factorize()[0]
   
df_tiers.head()

#Regression model for profit by tier (including intercept as a baseline)
tier_profit = sm.OLS(df_tiers[["GrossProfit"]], sm.add_constant(df_tiers[["CustomerTier"]])).fit()
tier_profit.summary()

connection.close()
