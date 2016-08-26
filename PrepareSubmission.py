# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:46:39 2016

@author: lbignell
"""

import shelve
import pandas
from sklearn.preprocessing import LabelEncoder
import numpy as np

def PrepareSubmission(modelpath, outputname):
    print("loading client table...")
    client_table_cleaned = pandas.read_csv('client_table_cleaned.csv',
                                           encoding = "ISO-8859-1")
    print("Done!")
    print("loading product table...")
    product_table = pandas.read_csv('product_table_extractPromMassPrice_BulkPrice.csv')
    print("Done!")
    #print("loading town_state...")
    #town_state = pandas.read_csv('town_state.csv')
    #print("Done!")
    print("loading test data...")
    test = pandas.read_csv('test.csv',
                           encoding = "ISO-8859-1")
    print("Done!")

    print("Merging data...")
    test = pandas.merge(product_table, test, on="Producto_ID")
    test = pandas.merge(client_table_cleaned, test, on="Cliente_ID")
    print("Done!")
    print("Encoding labels...")
    le = LabelEncoder()
    test.loc[:, "ClientNumber"] = pandas.Series(le.fit_transform(test.NombreCliente), 
                                                   index=test.index)

    test.loc[:, "IsProm"] = pandas.Series(le.fit_transform(test.IsProm),
                                             index=test.index)

    print("Done!")
    print("Cleaning up...")
    test.drop(["Cliente_ID", "NombreProducto", "NombreCliente", ], 
              inplace=True, axis=1)

    X = test[["Producto_ID", "IsProm", "Mass", "Price", "Semana",
              "Agencia_ID", "Canal_ID", "Ruta_SAK", "ClientNumber",
              "BulkPrice"]]
    X.fillna(value=-1, inplace=True)

    print("Running model...")
    sfile = shelve.open(modelpath, flag='r')
    predictions = np.array(sfile['model'].predict(X))
    sfile.close()
    print("Done!")
    print("Saving Output...")
    np.savetxt(outputname, predictions, delimiter=',',
               header='id,Demanda_uni_equil')
    print("Done!")
    return