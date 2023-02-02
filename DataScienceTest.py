from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px

pd_SELLDATA = pd.read_csv('D:\PYTHON\AULAS\TESTE_PS\Sheet1_SELLDATA.csv', skiprows=[0,1,2,3])
print('Dados de Vendas')
print(pd_SELLDATA.head(100))
print('-----------------------')

pd_CONSUMERDATA = pd.read_csv('D:\PYTHON\AULAS\TESTE_PS\Sheet3_CONSUMERDATA.csv')
print('Dados de Consumidores')
print(pd_CONSUMERDATA.head(10))
print('----------------------')

pd_STOREDATA = pd.read_csv('D:\PYTHON\AULAS\TESTE_PS\Sheet4_STOREDATA.csv')
print('Dados de Lojas')
print(pd_STOREDATA.head(10))
print('----------------------')

print('Dados de Produtos')
pd_PRODUCTDATA = pd.read_csv('D:\PYTHON\AULAS\TESTE_PS\Sheet2_PRODUCTDATA.csv')
print(pd_PRODUCTDATA.head(10))
print('----------------------')

print(pd_SELLDATA['UnitPrice'])
MEAN_VALUE = pd_SELLDATA['UnitPrice']
#MEAN_VALUE = pd_SELLDATA['UnitPrice'].mean()

print(len(pd_SELLDATA['ProductID'].unique()))

print(pd_SELLDATA['ProductID'].value_counts())
print(pd_SELLDATA['ProductID'].max())

#plt.plot(pd_SELLDATA['Quantity'],pd_SELLDATA['Date'])
#sns.lineplot(data=pd_SELLDATA, x="Quantity", y="2019")
df = pd_SELLDATA[['Date','Quantity','ProductID','UnitPrice','Discount','StoreID']]

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

df = df[(df['Year'] == 2019)]
#PARA PLOTAR PRA 1 SÓ PRODUTO
#df = df[(df['ProductID'] == '002d4ea7c04739c130bb74d7e7cd16943')]
df['Month'] = df['Date'].dt.month

#df['Quantity'] = 
#dg = df.groupby(pd.Grouper(key='date', freq='1M')).sum()
#dg.index = dg.index.strftime('%B')
#print(dg)

#df.groupby(df['Date'].dt.strftime('%B'))['Quantity'].sum().sort_values()
#df.set_index('Date',inplace=True)
#df.resample('M').sum().sort_values()
#df['Year'] = pd.DatetimeIndex(pd_SELLDATA['Date']).year

df['Months'] = df['Date'].apply(lambda x:x.strftime('%B'))
response_data_frame_QUANT = df.groupby('Month')['Quantity'].sum()
#response_data_frame_TOTAL['Revenue'] = (df['Quantity'] * df['UnitPrice'])


#response_data_frame_QUANT = df.groupby('Month')['Quantity'].sum()

#df 
print('  ')
print('Tabelas de faturamento mensal')
print('-----------')
revenue = pd_SELLDATA[['Date','Quantity','ProductID','UnitPrice','Discount']]
revenue['totalprice'] = revenue['UnitPrice'] * revenue['Quantity']





df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

print('---------------')
print('DF FINAL')
print(df)




revenue['Date'] = pd.to_datetime(revenue['Date'])
revenue['Year'] = revenue['Date'].dt.year

revenue = revenue[(revenue['Year'] == 2019)]

#revenue['Amount'] = (revenue['UnitPrice'] revenue['Discount']) 
#evenue['Amount']
#revenue['revenue'] = revenue['Quantity']*revenue['Amount']
print(revenue)
print(' ')

print('--------------')
print('Vendas totais por mês em 2019:')
print(response_data_frame_QUANT)


#TROCANDO AS VIRGULAS DOS VALORES POR PONTOS PARA MANIPULAR OS DADOS
df['UnitPrice'] = df['UnitPrice'].apply(lambda x: x.replace(',','.'))
df['Discount'] = df['Discount'].apply(lambda x: x.replace(',','.'))

df['UnitPrice'] = df['UnitPrice'].astype(float)
df['Discount'] = df['Discount'].astype(float)

#if df['UnitPrice'].index > 1000:
#    df['UnitPrice'].index = df['UnitPrice'].mean()



#OBTENCAO VALORES FATURAMENTO 
df['Fatura'] = (df['UnitPrice'] * df['Quantity'])
df['Fatura Liq'] = (df['Fatura'] - df['Discount'])

FAT_MENSAL = df.groupby('Month')['Fatura Liq'].sum()

#DESCRICAO DOS DADOS COLUNA FATURA
print(df['Fatura'].describe())
print(df['Fatura Liq'].describe())
print(df)

ESCOLHA = df.groupby('StoreID')['Fatura Liq'].sum()
print(ESCOLHA)
'''
print('----------------')
print('   ')
print('DADOS DF')
print(df)
print('  ')
print('----------')
print('PRODUTOS AGRUPADOS POR MES (QUANTIDADE)')
df_QUANT = df.groupby('Month')['Quantity'].sum()
print(df_QUANT)

print('  ')
print('----------')
print('PRODUTOS AGRUPADOS POR MES (DISCONTO)')
df_DISCOUNT = df.groupby('ProductID')['Discount'].sum()
print(df_DISCOUNT)

print('  ')
print('----------')
print('PRODUTOS AGRUPADOS POR MES (FATURAM)')
df_UPRICE = df.groupby('ProductID')['UnitPrice'].sum()
print(df_UPRICE)
'''
#f['Month'] = pd.DatetimeIndex(pd_SELLDATA['Date']).month

#grafico1 = plt.plot(df['Month'], df['Quantity'])
#plt.show()

plt.plot(response_data_frame_QUANT)
plt.show()

print(FAT_MENSAL)
print(FAT_MENSAL.describe())

ESCOLHA_LOJA = pd_CONSUMERDATA

