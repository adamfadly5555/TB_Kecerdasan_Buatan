
# 🧠 Proyek Kecerdasan Buatan: Prediksi Kesehatan Berdasarkan Gaya Hidup

## 📄 Deskripsi
Proyek ini bertujuan untuk memprediksi status kesehatan seseorang berdasarkan kebiasaan seperti merokok, begadang, dan aktivitas olahraga, menggunakan beberapa algoritma machine learning seperti KNN, Decision Tree, Naive Bayes, dan SVM.

---

## 🔧 Langkah-langkah Pengerjaan

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

### 3. Visualisasi Awal Data
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Merokok', data=df)
plt.title('Distribusi Status Merokok')
plt.show()
```

### 4. Pembersihan Data
```python
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
```

### 5. Encoding Label dan Standardisasi
```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

numerik_fitur = ['Usia', 'Aktivitas_Begadang', 'Aktivitas_Olahraga']
scaler = StandardScaler()
df_encoded[numerik_fitur] = scaler.fit_transform(df_encoded[numerik_fitur])
```

### 6. Visualisasi Korelasi
```python
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi antar fitur")
plt.show()
```

### 7. Split Data (Train-Test)
```python
from sklearn.model_selection import train_test_split

X = df_encoded.drop('Hasil', axis=1)
y = df_encoded['Hasil']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 8. Modeling dengan KNN
```python
from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
```

### 9. Evaluasi Model
```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model_knn.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 10. Visualisasi PCA
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_train_pca, y_train)
```

### 11. Pembandingan Beberapa Model
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: Akurasi = {acc:.2f}")
```

---

## 📦 Library yang Dibutuhkan
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## 📝 Catatan
- Dataset berada di Google Drive (`predic_tabel.csv`)
- Proyek ini dapat dijalankan di Google Colab untuk kemudahan penggunaan.
