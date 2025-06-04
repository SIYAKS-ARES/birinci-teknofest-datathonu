# Stacked Sinir Ağı (Stacked NN)

Bu çözüm, farklı modellerin güçlü yönlerini bir araya getirerek oluşturulmuş bir **stacked ensemble** yaklaşımıdır. Özellikle sızıntısız (leak-free) veri işleme, OOF tahminlerin tekrar kullanımı ve outlier sınıflandırması gibi ileri düzey teknikleri barındırmaktadır.

---

## 🚧 Stratejiye Genel Bakış

- Farklı modeller oluşturuldu ve bu modellerin **out-of-fold (OOF)** tahminleri daha sonra başka modeller için giriş (feature) olarak kullanıldı.
- Ensemble stratejisinde temel fikir: Farklı yapıdaki modellerin hatalarının birbiriyle örtüşmemesi ve bu sayede genel hata oranının düşmesidir.
- İki farklı final gönderimi yapıldı:
  - İlki: Ridge ile birleştirilmiş ensemble
  - İkincisi (birinciliği getiren): Genişletilmiş sinir ağı (NN) modeli

---

## 🧪 Veri İşleme ve Çapraz Doğrulama

- **20-fold çapraz doğrulama** kullanıldı. Bu sayede modelin genellenebilirliği artırıldı.
- Kategorik değişkenler için **target encoding** uygulandı.
  - Ortalama yerine **medyan** kullanıldı.
  - Her fold'da **yeniden hesaplandı**, böylece veri sızıntısı (data leakage) önlendi.

---

## 🎯 Outlier (Aykırı Değer) Sınıflandırması

Bazı modellerin fiyatları doğru tahmin edememesi outlier'lardan kaynaklanıyordu. Bu nedenle önce outlier’lar sınıflandırıldı ve bu bilgi diğer modellere eklendi.

```python
def bin_price(data):
    df = data.copy()
    Q1 = np.percentile(df['price'], 25)
    Q3 = np.percentile(df['price'], 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['price_bin'] = (df['price'] < upper_bound).astype(int)
    return df

	•	price_bin özelliği CatBoostClassifier ile tahmin edildi ve diğer modellere özellik olarak eklendi.

⸻

🧠 Kullanılan Ana Modeller

1. CatBoostClassifier (Outlier sınıflayıcı)

cat_params2 = {
    'early_stopping_rounds': 25,
    'use_best_model': True,
    "verbose": False,
    'cat_features': cat_cols,
    'min_data_in_leaf': 16, 
    'learning_rate': 0.0335, 
    'random_strength': 11.66, 
    'l2_leaf_reg': 17.70, 
    'max_depth': 10, 
    'subsample': 0.947, 
    'border_count': 130, 
    'bagging_temperature': 24.03
}

	•	CatBoostClassifier tahminleri başka modellere girdi (feature) olarak verildi.

⸻

2. LightGBM (LGBM5)

lgb_params = {
    'verbose': -1,
    'early_stopping_rounds': 25,
    'loss_function': "RMSE",
    'n_estimators': 2000, 
    'max_bin': 30000,
}

	•	Kategorik değişkenler label encoding ile işlendi.
	•	Nadir kategoriler “rare” olarak gruplanarak overfitting azaltıldı.

⸻

3. SVR (Support Vector Regression)
	•	rbf çekirdeği kullanıldı.
	•	Özellikle orta-düzey fiyatlar için stabil performans verdi.
	•	SVR OOF tahminleri sinir ağı modeline eklendi.

⸻

4. AutoGluon (FastAI Modeli)

predictor = TabularPredictor(label='price',
                             eval_metric='rmse',
                             problem_type="regression").fit(
    X_train,
    pseudo_data=data_original, 
    num_bag_folds=10,
    num_bag_sets=2,
    time_limit=1800,
    included_model_types=['FASTAI'], 
    keep_only_best=True,
    presets="best_quality"
)

	•	Nested CV uygulandı.
	•	Not: AutoGluon’un fit() fonksiyonu orijinal veriyi kullanmaz, bu yüzden fit_pseudolabel() kullanılması önerilir.

⸻

🔄 Ensemble Yapısı: Genişletilmiş Sinir Ağı

Finalde kullanılan stacked NN, aşağıdaki 4 modelin OOF tahminlerini özellik olarak aldı:
	1.	SVR tahminleri
	2.	CatBoostClassifier tahminleri
	3.	LGBM5 tahminleri
	4.	XGB tahminleri (kaynağı unutulmuş)

	•	Bu yapı, klasik ensemble yaklaşımından daha iyi sonuç verdi.
	•	NN modeli değişikliklere karşı robust olduğu için tercih edildi.

⸻

📊 Sonuçlar

Model	CV RMSE
Ensemble (Ridge)	72300
Stacked NN	72468 (ancak LB’de daha başarılı)

	•	LB skoru daha iyi olduğu için stacked NN birinciliği getirdi.
	•	OOF tahminleri özellik olarak kullanmak, başarıyı önemli ölçüde artırdı.

⸻

🧩 Ek Bilgiler ve Notlar
	•	XGBoost, finalde kazara dahil edildi ancak katkısı sınırlıydı.
	•	Chris Deotte’un feature engineering fikirleri bu veri setinde işe yaramadı.
	•	AutoML üzerine yapılan tartışmalar (özellikle AutoGluon) çok faydalı oldu.


📌 Öneri Niteliğinde Kullanım

Yukarıdaki yöntemler:
	•	Katmanlı (stacked) modelleme
	•	Sızıntısız hedef kodlama
	•	OOF tahminlerin yeniden kullanımı
	•	Outlier sınıflandırması

gibi teknikleri kullanarak benzer regresyon problemlerinde yüksek performans elde edilebilir.
