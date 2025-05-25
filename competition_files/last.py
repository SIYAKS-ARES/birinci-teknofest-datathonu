import pandas as pd

# Dosyaları oku
train = pd.read_csv("train_new.csv")
test = pd.read_csv("test_new.csv")

# Birleştirerek aynı işlemleri yapabilmek için:
combined = pd.concat([train.drop(columns=["price"]), test], axis=0)

# Sayısal sütunlardaki eksik değerleri ortalama ile doldur
for col in ["horsepower", "engine_liter"]:
    combined[col] = combined[col].fillna(combined[col].mean())

# Kategorik eksikleri doldur (örnek: 'fuel_type')
for col in ["fuel_type", "transmission", "color_int"]:
    combined[col] = combined[col].fillna("Unknown")

'''# train ve test'i ayır
train_cleaned = combined.iloc[:len(train)]
test_cleaned = combined.iloc[len(train):]

# train'e tekrar 'price' sütunu ekle
train_cleaned["price"] = train["price"]
'''

# Doğru şekilde kopya alın
train_cleaned = combined.iloc[:len(train)].copy()
test_cleaned = combined.iloc[len(train):].copy()

# Güvenli bir şekilde price sütununu ekle
train_cleaned.loc[:, "price"] = train["price"]
