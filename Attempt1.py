# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 23:03:29 2016

@author: lbignell
"""
numrows = 10000

print("Importing packages...")
import numpy as np
import pandas as pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
import shelve
import time

def RMSLE(pred, actual):
    if len(pred)!=len(actual):
        print("RMSLE ERROR: Incompatible sizes!")
        return
    return (sum((np.log(pred + 1) - np.log(actual + 1))**2)/len(predictions))**0.5

print("Done!")

print("loading client table...")
client_table_cleaned = pandas.read_csv('client_table_cleaned.csv',
                                       encoding = "ISO-8859-1")
print("Done!")
print("loading product table...")
product_table = pandas.read_csv('product_table_extractPromMassPrice.csv')
print("Done!")
print("loading town_state...")
town_state = pandas.read_csv('town_state.csv')
print("Done!")
#print("loading test data...")
#test = pandas.read_csv('test.csv')
#print("Done!")
print("loading training data...")
#Use random training data instead.
#train = pandas.read_csv('train.csv', nrows=numrows)
train = pandas.read_csv("train_{0}Krandom.csv".format(int(numrows/1000)),
                        encoding = "ISO-8859-1")
print("Done!")

print("Merging data...")
train_first = pandas.merge(product_table, train, on="Producto_ID")
train_first = pandas.merge(client_table_cleaned, train_first, on="Cliente_ID")
print("Done!")
print("Encoding labels...")
le = LabelEncoder()
train_first.loc[:, "ClientNumber"] = pandas.Series(le.fit_transform(train_first.NombreCliente), 
                                                   index=train_first.index)
train_first.loc[:, "IsProm"] = pandas.Series(le.fit_transform(train_first.IsProm),
                                             index=train_first.index)
print("Done!")
print("Cleaning up...")
train_first.loc[:, "BulkPrice"] = train_first.Venta_hoy/train_first.Venta_uni_hoy
train_first.drop(["Cliente_ID", "NombreProducto", "NombreCliente", 
                  "Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", 
                  "Dev_proxima"], inplace=True, axis=1)

X_train = train_first[["Producto_ID", "IsProm", "Mass", "Price", "Semana",
                       "Agencia_ID", "Canal_ID", "Ruta_SAK", "ClientNumber",
                       "BulkPrice"]]
X_train.fillna(value=-1, inplace=True)
y_train = train_first["Demanda_uni_equil"]
print("Max semana: {0}".format(max(X_train["Semana"])))
print("Done!")

print("Instanciating and fitting model...")
model = RandomForestRegressor(n_estimators=100, n_jobs=1)
model.fit(X_train, y_train)
print("Done!")

print("Cross-validation...")
scores = cross_val_score(model, X_train, y_train, cv=10, scoring="mean_squared_error")
print("Done!")

#Initial test with training data...
print("Checking predictions...")
predictions = np.array(model.predict(X_train))
rmse = np.sqrt(np.mean((np.array(y_train.values) - predictions)**2))
imp = sorted(zip(X_train.columns, model.feature_importances_), key=lambda tup: tup[1], reverse=True)
print("Done!")

print("RMSE: " + str(rmse))
print("10 Most Important Variables:" + str(imp[:10]))

print("Scores:" + str((-1*scores)**0.5))

print("RMSLE: " + str(RMSLE(predictions, np.array(y_train.values))))

print("Saving model...")
f = shelve.open('FirstModel_random_{0}rows_{1}.shelf'.format(numrows, 
                int(time.time())))
f['model'] = model
f.close()
print("Done!")

print("10 Most Important Variables:" + str(imp[:10]))

print("Scores:" + str((-1*scores)**0.5))
