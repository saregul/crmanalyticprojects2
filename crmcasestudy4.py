#BG-NBD ve Gamma-Gamma ile CLTV Tahmini
#İngiltere Merkezli Perakende Şirketi CLTV Tahmini

################################################################
#İŞ PROBLEMİ
#İngiltere merkezli perakende şirketi satış ve pazarlama
#faaliyetleri için roadmap belirlemek istemektedir. Şirketin
#orta uzun vadeli plan yapabilmesi için var olan müşterilerin
#gelecekte şirkete sağlayacakları potansiyel değerin
#tahmin edilmesi gerekmektedir.

import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df_ = pd.read_excel("venv/Lib/site-packages/zipp/online_retail_II.xlsx", sheet_name=2010-2011) #2010-2011 arası verinin okutulması
df = df_.copy() #dataframe kopyası
df.describe() #betimsel istatistikler
df.info #veri hakkında bilgi
df.isnull().sum() #eksik gözlem sayısı
df.dropna(inplace=True) #eksik gözlem silme
df.head()

import sklearn
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter



df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)


#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df['frequency'] = df.groupby('Customer ID')['Invoice'].nunique()
cltv_df['recency'] = (today_date - df.groupby('Customer ID')['InvoiceDate'].max()).dt.days
cltv_df['T'] = (today_date - df.groupby('Customer ID')['InvoiceDate'].min()).dt.days


cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 0)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T" > 0]


bgf = BetaGeoFitter(penalizer_coef=0.001)

cltv_df['frequency'].fillna(0, inplace=True)
cltv_df['recency'].fillna(0, inplace=True)
cltv_df['T'].fillna(0, inplace=True)

cltv_df['frequency'] = cltv_df['frequency'].astype(int)
cltv_df['recency'] = cltv_df['recency'].astype(int)
cltv_df['T'] = cltv_df['T'].astype(int)


bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


####################################
# 2. BG-NBD Modelinin Kurulması
####################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

cltv_df['frequency'].fillna(0, inplace=True)
cltv_df['recency'].fillna(0, inplace=True)
cltv_df['T'].fillna(0, inplace=True)

cltv_df['frequency'] = cltv_df['frequency'].astype(int)
cltv_df['recency'] = cltv_df['recency'].astype(int)
cltv_df['T'] = cltv_df['T'].astype(int)


bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

# 1 ay içerisinde en çok satın alma beklediğimiz 10 müşteri kimdir ?

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# 1 ay içerisinde en çok satın alma beklediğimiz 10 müşteri kimdir ?

bgf.predict(4*12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

#2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv_df_segment"] = pd.qcut(cltv_df["cltv_df"], 4, labels=("D", "C", "B", "A"))
cltv_df.head()
