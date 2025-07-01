
# 🧠 Tugas Besar Kecerdasan Buatan: Prediksi Penyakit Paru-paru Berdasarkan Data Kebiasaan Harian Dengan Metode K-Nearest Neighboor

## 📄 Deskripsi
Tujuan utama dari proyek ini adalah membangun dan mengimplementasikan sebuah sistem klasifikasi berbasis kecerdasan buatan yang mampu memprediksi risiko penyakit paru-paru pada individu berdasarkan data gejala dan faktor risiko seperti kebiasaan merokok, usia, dan pola aktivitas. Sistem ini dikembangkan menggunakan algoritma K-Nearest Neighbor (KNN) karena algoritma ini sederhana, efektif, serta memiliki kemampuan tinggi dalam menangani data dengan banyak variabel dan memberikan hasil klasifikasi yang akurat. Dengan adanya sistem ini, diharapkan proses deteksi dini penyakit paru-paru dapat dilakukan secara otomatis dan efisien, sehingga mendukung pengambilan keputusan medis yang lebih cepat dan tepat.

---

##  Langkah-langkah Pengerjaan

### 1. Mount Google Drive & Akses Dataset
```python
from google.colab import drive
drive.mount('/content/drive')
dataset_path = "/content/drive/MyDrive/DATASET/predic_tabel.csv"
```

### 2. Load Dataset dan Eksplorasi Awal
```python
import pandas as pd
df = pd.read_csv(dataset_path)

print("Shape:", df.shape)
df.info()
df.describe()
```

### 3. Eksplorasi Awal Data
```python
# Ukuran data
print("Shape:", df.shape)

df.info()

# Statistik deskriptif
df.describe()
```

### 4. Pembersihan Data
```python
print(df.dtypes)
```

### 5. Encoding Label dan Standardisasi
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Merokok', data=df)
plt.title('Distribusi Status Merokok')
plt.xlabel('Status Merokok')
plt.ylabel('Jumlah')
plt.show()
```

### 6. Visualisasi Korelasi
```python
sprint(df.columns)
```

### 7. Split Data (Train-Test)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisasi distribusi fitur kategorik dari dataset
plt.figure(figsize=(12, 4))

# Barplot untuk kolom Merokok
plt.subplot(1, 3, 1)
sns.countplot(x='Merokok', data=df)
plt.title('Distribusi Merokok')

# Barplot untuk kolom Aktivitas_Begadang
plt.subplot(1, 3, 2)
sns.countplot(x='Aktivitas_Begadang', data=df)
plt.title('Distribusi Begadang')

# Barplot untuk kolom Hasil
plt.subplot(1, 3, 3)
sns.countplot(x='Hasil', data=df)
plt.title('Distribusi Hasil')

plt.tight_layout()
plt.show()
```

### 8. Modeling dengan KNN
```python
print(df['Hasil'].value_counts(normalize=True))
```

### 9. Evaluasi Model
```python
import pandas as pd

print(pd.crosstab(df['Merokok'], df['Hasil'], normalize='index'))

print(pd.crosstab(df['Usia'], df['Hasil'], normalize='index'))

```

### 10. Visualisasi PCA
```python
print(df.isnull().sum())

print(df.duplicated().sum())
```

### 11. Pembandingan Beberapa Model
```python
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
```
### 11. Pembandingan Beberapa Model
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

print(df_encoded)


```
### 11. Pembandingan Beberapa Model
```python
from sklearn.preprocessing import StandardScaler

numerik_fitur = ['Usia', 'Aktivitas_Begadang', 'Aktivitas_Olahraga']

scaler = StandardScaler()
df_encoded[numerik_fitur] = scaler.fit_transform(df_encoded[numerik_fitur])

print("Data setelah standardisasi:")
print(df_encoded.head())


```
### 11. Pembandingan Beberapa Model
```python
import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi antar fitur numerik")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


```
### 11. Pembandingan Beberapa Model
```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Hasil', axis=1)
y = df_encoded['Hasil']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:\n", X_train)
print("y_train:\n", y_train)


```
### 11. Pembandingan Beberapa Model
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)


```
### 11. Pembandingan Beberapa Model
```python

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred_knn = model_knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - KNN")
plt.show()

```
### 11. Pembandingan Beberapa Model
```python

# Prediksi data test
y_pred = model.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nKlasifikasi:\n", classification_report(y_test, y_pred))

```
### 11. Pembandingan Beberapa Model
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduksi dimensi ke 2 komponen
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Latih ulang model KNN di ruang PCA
knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_train_pca, y_train)

# Prediksi data test di ruang PCA
y_pred_pca = knn_2d.predict(X_test_pca)

# Buat visualisasi
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_pca, cmap='viridis', alpha=0.7)
plt.title("Visualisasi Klasifikasi KNN dalam 2D (PCA)")
plt.xlabel("Komponen Utama 1")
plt.ylabel("Komponen Utama 2")
legend1 = plt.legend(*scatter.legend_elements(), title="Kelas")
plt.gca().add_artist(legend1)
plt.grid(True)
plt.show()

```
### 11. Pembandingan Beberapa Model
```python

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("Akurasi KNN:", accuracy_score(y_test, y_knn_pred))
print("\nKlasifikasi KNN:\n", classification_report(y_test, y_knn_pred))

```
### 11. Pembandingan Beberapa Model
```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_nb_pred = nb.predict(X_test)

print("Akurasi Naive Bayes:", accuracy_score(y_test, y_nb_pred))
print("\nKlasifikasi Naive Bayes:\n", classification_report(y_test, y_nb_pred))


```
### 11. Pembandingan Beberapa Model
```python
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
y_svm_pred = svm.predict(X_test)

print("Akurasi SVM:", accuracy_score(y_test, y_svm_pred))
print("\nKlasifikasi SVM:\n", classification_report(y_test, y_svm_pred))


```
### 11. Pembandingan Beberapa Model
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results.append({"Model": name, "Akurasi": round(acc, 2)})

df_results = pd.DataFrame(results)
print(df_results)


```
## Kesimpulan
Proyek ini berhasil membangun sebuah sistem prediksi penyakit paru-paru berbasis algoritma K-Nearest Neighbor (KNN) dengan menggunakan data dari platform Kaggle. Proses pembangunan sistem meliputi pembersihan data, transformasi fitur, pelatihan model, serta evaluasi kinerja. Meskipun seluruh pipeline sudah dijalankan dengan benar, hasil akhir model KNN menunjukkan akurasi yang masih rendah, yaitu sekitar 38,9%. Ini berarti model belum dapat secara andal digunakan untuk klasifikasi risiko penyakit paru-paru. Penyebab utamanya kemungkinan berasal dari kualitas dataset, ketidakseimbangan kelas, dan kompleksitas data yang tidak bisa ditangani secara optimal oleh KNN tanpa teknik tambahan.
