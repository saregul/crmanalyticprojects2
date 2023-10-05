######################################################################
#FLO BG-NBD ve Gamma Gamma ile CLTV Tahmini
import pandas as pd

############################################################################################################
#İŞ PROBLEMİ

#FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan yapabilmesi için
#var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

############################################################################################################
import pandas as pd
import datetime as dt
import sklearn
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None
pd.set_option('display.width', 1000)


# 1. Omnichannel.csv verisini okuyunuz. Dataframe'in kopyasını oluşturunuz.

df_ = pd.read_csv("venv/Lib/site-packages/zipp/flo_data_20k.csv")
df = df_.copy()
df.head()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


#"order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"
#değişkenlerinin aykırı değerleri var ise baskılayınız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

# Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

#Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


#CLTV Veri Yapısının Oluşturulması

#Veri setindeki en son  alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)


#customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[ns]')) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[ns]')) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()


# BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 Aylık CLTV'nin hesaplanması

#BG/NBD modelini kurunuz.
cltv_df = cltv_df[(cltv_df['frequency'] > 0)]
cltv_df["recency"] = cltv_df["recency_cltv_weekly"] / 7
cltv_df["T"] = cltv_df["T_weekly"] / 7

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

#6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

cltv_df[["exp_sales_3_month", "exp_sales_6_month"]]

# 3. ve 6.aydaki en çok satan alım gerçekleştirecek 10 kişiyi inceleyiniz. Fark var mı ?

cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]


# Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak
# cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()


# 6 aylık CLTV hesaplayınız ve CLTV ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv


#CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv", ascending=False)[:10]


#GÖREV: CLTV'ye göre segmentlerin oluşturulması

#6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi 4 gruba(segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
#cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=("D", "C", "B", "A"))
cltv_df.head()


#Tüm süreci fonksiyonlaştırınız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df = df[~(df["customer_value_total"] == 0) | (df["order_num_total"] == 0)]
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)



