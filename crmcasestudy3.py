# İngiltere Merkezli bir Perakende Şirketi için RFM Analizi

#İŞ PROBLEMİ

#İngiltere merkezli perakende şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama
#stratejileri belirlemek istemektedir.
#Ortak davranışlar sergileyen müşteri segmentleri özelinde pazarlama çalışmaları yapmanın gelir artışı sağlayacağını
#düşünmektedir.

#Veri Setini Okuma

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

#fonksiyonları kullanmak için tanımladık.


df_ = pd.read_excel("venv/Lib/site-packages/zipp/online_retail_II.xlsx", sheet_name=2010-2011) #2010-2011 arası verinin okutulması
df = df_.copy() #dataframe kopyası
df.describe() #betimsel istatistikler
df.info #veri hakkında bilgi
df.isnull().sum() #eksik gözlem sayısı
df.dropna(inplace=True) #eksik gözlem silme
df.head()
df = df[~df["Invoice"].str.contains("C", na=False)] #invoice değerini başında C olan yani iade olan ürünler dışındaki ürünleri bize getir.
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

today_date = dt.datetime(2011, 12, 11)

df["TotalPrice"] = df["Quantity"] * df["Price"] #her müşteriden gelen toplam maliyet

cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


cltv_df['frequency'] = df.groupby('Customer ID')['Invoice'].nunique() #eşsiz değer saysı
cltv_df['recency'] = (today_date - df.groupby('Customer ID')['InvoiceDate'].max()).dt.days
cltv_df['T'] = (today_date - df.groupby('Customer ID')['InvoiceDate'].min()).dt.days

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df['frequency'] > 0)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T" > 0]

####################################################################
#GÖREV 2: RFM Metriklerinin Hesaplanması
###################################################################

#Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["InvoiceDate"].max() #2021-05-30
analysis_date = dt.datetime(2011,12,11) #2021-06-01

#customer_id, recency, frequency ve monetary değerlerinin yer aldığı yeni bir rfm dataframe oluşturunuz.

rfm = pd.DataFrame()
rfm["customer_id"] = df["Customer_ID"]
rfm["recency"] = (analysis_date - df["InvoiceDate"]).astype('timedelta64[us]')
rfm["frequency"] = df["Quantity"]
rfm["monetary"] = df["TotalPrice"]

#Recency, Frequency ve Monetary metriklerinin qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
#bu skorlara recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

#recency_score ve frequency_score'u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()


#Oluşturulan RF skorları için segment tanımlamaları yapınız
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2][4-5]': 'cant_loose_them',
    r'[3][1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm.head()

# Segmentlerin recency, frequency ve monetary ortalamalarını inceleyiniz.
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
#                                   recency        frequency        monetary
#                                        mean  count      mean  count     mean  count
#segment
#about_to_sleep      133 days 20:09:38.843299  28564      1.72  28564     7.32  28564
#at_risk             270 days 17:42:52.858125  69780      8.38  69780    17.90  69780
#cant_loose_them     274 days 08:25:49.661417  25843     46.18  25843    73.67  25843
#champions            19 days 23:10:31.330049  28420     23.77  28420    42.57  28420
#hibernating         278 days 14:34:23.811847  63523      1.76  63523     8.22  63523
#loyal_customers     100 days 09:10:39.492163  68014     25.63  68014    39.04  68014
#need_attention      132 days 20:28:14.334130  15048      5.09  15048    15.74  15048
#new_customers        19 days 14:52:53.221864  18660      1.00  18660     4.07  18660
#potential_loyalists  41 days 19:42:26.865807  66084      3.63  66084    11.57  66084
#promising            61 days 06:06:01.058140  13949      1.00  13949     6.10  13949

#"Loyal Customers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.

loyal_customer_ids = rfm[rfm["segment"].isin(["loyal_customers"])]["customer_id"]
loyal_customer_ids.to_excel("yeni_marka_hedef_musteri_id.xlsx", index=False)
loyal_customer_ids.shape
