######################################################
# FLO RFM Analizi ile Müşteri Segmentasyonu
import pandas as pd

######################################################
## İş Problemi
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacaktır.
######################################################


#Adım 1: flo_data_20k.csv verisini okuyunuz. Dataframe'in kopyasını oluşturunuz.
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

df_ = pd.read_csv("venv/Lib/site-packages/zipp/flo_data_20k.csv")
df = df_.copy() #verinin kopyası

df["order_channel"].value_counts() #alışveriş yapılan kanallar(cihaz veya elden)
df["last_order_channel"].value_counts()

#Adım 2

df_.head(10)  #ilk 10 veri
df.columns
df.shape
df_.describe().T #betimsel istatistik
df_.isnull().sum() #boş değer toplamı
df.info()

#değişken tipi sorgulamasına bak

#Adım 3
# Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total"] = df["order_num_total_ever_online"] * df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] * df["customer_value_total_ever_online"] #hem online hem de offline alışveriş yapan müşteri sayısı
df.head()

#Adım 4
#Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

#Adım 5
# Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısı ve toplam harcamaların dağılımına bakınız.

df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})


#Adım 6
# En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("customer_value_total", ascending=False)[:10]


#Adım 7
#En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("order_num_total", ascending=False)[:10]


#Adım 8
#Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] * dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] * dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

####################################################################
#GÖREV 2: RFM Metriklerinin Hesaplanması
####################################################################

#Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["last_order_date"].max() #2021-05-30
analysis_date = dt.datetime(2021,6,1) #2021-06-01

#customer_id, recency, frequency ve monetary değerlerinin yer aldığı yeni bir rfm dataframe oluşturunuz.

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[us]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

# GÖREV 3: RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)

#Recency, Frequency ve Monetary metriklerinin qcut yardımı ile 1-5 arasında skorlara çevrilmesi ve
#bu skorlara recency_score, frequency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()


#recency_score ve frequency_score'u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()


# recency_score, frequency_score ve monetary_score'u tek bir değişken olarak ifade edilmesi ve RFM_SCORE olarak kaydedilmesi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))
rfm.head(10)



#######################################################################################
# GÖREV 4: Oluşturulan RFM skorların daha açıklanabilir olması için segment tanımlamave tanımlanan seg_map yardımı ile
# RF_SCORE'u segmentlere çevirin.

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

#GÖREV 5
# Segmentlerin recency, frequency ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                                       mean count      mean count      mean count
#segment
#about_to_sleep      113 days 18:50:36.464088  1629      1.40  1629  32429.79  1629
#at_risk             242 days 06:36:33.316195  3112      3.98  3112  82913.47  3112
#cant_loose_them     233 days 20:00:11.812961  1219     17.49  1219 319293.73  1219
#champions            17 days 02:08:39.122257  1914     14.14  1914 341148.53  1914
#hibernating         247 days 22:47:16.847946  3604      1.39  3604  33690.59  3604
#loyal_customers      83 days 02:18:43.996431  3363     12.97  3363 265228.52  3363
#need_attention      113 days 08:30:33.707865   801      2.90   801  65652.73   801
#new_customers        17 days 22:01:24.705882   680      1.00   680  30327.28   680
#potential_loyalists  37 days 03:25:38.709677  2976      2.42  2976  79301.76  2976
#promising            58 days 22:06:29.489953   647      1.00   647  28636.78   647#



# Verilen 2 case için ilgili profildeki müşterileri bulunuz ve müşteri id'lerini csv'ye kaydediniz.

#a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstündedir.
#Bu nedenle bu markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçilmek isteniliyor.
#Bu müşterilerin sadık ve kadın kategorisinden alışveriş yapan kişiler olması planlandı.Müşterilerin id numaralarını
#csv dosyasına yeni_marka_hedef_musteri_id.cvs olarak kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_musteri_id.cvs", index=False)
cust_ids.shape

rfm.head()


#b. Erkek ve çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden
#olan ama uzun süredir alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyası olarak kaydedin.

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose_them", "at_risk", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)



############################################################################################
def create_rfm(dataframe):
    #Veriyi Hazırlama
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
















