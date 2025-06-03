# Proyek Klasifikasi Gambar: Klasifikasi 10 Jenis Hewan

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan model _deep learning_ yang mampu mengklasifikasikan gambar ke dalam 10 kelas hewan yang berbeda. Dataset yang digunakan adalah "Animals-10" yang bersumber dari Kaggle. Model yang dikembangkan menggunakan arsitektur _hybrid_, menggabungkan _transfer learning_ dengan _base model_ MobileNetV2 dan penambahan layer-layer `Conv2D`, `BatchNormalization`, serta `MaxPooling2D` secara eksplisit untuk memenuhi kriteria _submission_ dan meningkatkan performa klasifikasi.

## Dataset

* **Sumber:** Kaggle - Animals-10 Dataset [https://www.kaggle.com/datasets/alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
* **Deskripsi:** Dataset ini berisi lebih dari 26.000 gambar yang terbagi ke dalam 10 kelas hewan: `cane` (anjing), `cavallo` (kuda), `elefante` (gajah), `farfalla` (kupu-kupu), `gallina` (ayam), `gatto` (kucing), `mucca` (sapi), `pecora` (domba), `ragno` (laba-laba), dan `scoiattolo` (tupai).
* **Pembagian Data:** Dataset dibagi menjadi _training set_ (70%), _validation set_ (15%), dan _test set_ (15%).
    * Jumlah Sampel Training: **[18336]**
    * Jumlah Sampel Validasi: **[3904]**
    * Jumlah Sampel Test: **[3968]**

## Arsitektur Model

Model yang digunakan adalah model _hybrid_ yang dibangun secara `Sequential` dengan detail sebagai berikut:
1.  **Input Layer:** Menerima gambar dengan ukuran **[(150, 150, 3)]**.
2.  **Preprocessing Layer:** Menggunakan layer `Lambda` untuk menerapkan fungsi `preprocess_input` dari MobileNetV2.
3.  **Base Model:** Menggunakan MobileNetV2 (_pretrained_ pada ImageNet) dengan _classifier head_ bawaan dihilangkan (`include_top=False`). Bobot dari _base model_ ini dibekukan (`trainable=False`) selama tahap awal _training_.
4.  **Layer Konvolusi Eksplisit:** Untuk memenuhi kriteria, ditambahkan layer-layer berikut setelah _base model_:
    * `Conv2D` (64 filter, kernel 3x3, aktivasi ReLU, padding 'same')
    * `BatchNormalization`
    * `MaxPooling2D` (2x2)
    * `Conv2D` (128 filter, kernel 3x3, aktivasi ReLU, padding 'same')
    * `BatchNormalization`
    * `MaxPooling2D` (2x2)
5.  **Classifier Head:**
    * `GlobalAveragePooling2D`
    * `Dense` (128 unit, aktivasi ReLU)
    * `Dropout` (0.5)
    * `Dense` (64 unit, aktivasi ReLU)
    * `Dropout` (0.3)
    * `Dense` (_output layer_ dengan `num_classes` unit (10 kelas) dan aktivasi `softmax`)

## Proses Training

* **Optimizer:** Adam dengan _learning rate_ awal **[1e-4]**.
* **Loss Function:** `sparse_categorical_crossentropy` (karena `label_mode='int'`).
* **Metrics:** `accuracy`.
* **Callbacks:**
    * `ModelCheckpoint`: Menyimpan model dengan `val_accuracy` terbaik.
    * `EarlyStopping`: Menghentikan _training_ jika `val_accuracy` tidak meningkat setelah **[Isi dengan patience ES Anda, misal 15]** epoch, dengan `restore_best_weights=True`.
    * `ReduceLROnPlateau`: Mengurangi _learning rate_ jika `val_loss` tidak membaik setelah **[Isi dengan patience RLP Anda, misal 7]** epoch.
* **Jumlah Epoch:** Training dilakukan selama **[25 hingga dihentikan EarlyStopping]** epoch.

## Hasil Model

* **Akurasi Validasi Terbaik:** **[misal 93.80%]**
* **Akurasi Test Set:** **[93.86%]**
* **Loss Test Set:** **[0.2077]**

Model berhasil mencapai akurasi di atas 85% pada _training set_, _validation set_, dan _test set_.

## Format Model yang Disimpan

Model disimpan dalam tiga format:
1.  **TensorFlow SavedModel**
2.  **TensorFlow Lite**
3.  **TensorFlow.js**

## Cara Menjalankan Notebook

1.  Pastikan _environment_ Python sudah terinstal beserta _library_ yang ada di `requirements.txt`.
2.  Unggah file `kaggle.json` Anda saat diminta oleh _notebook_ untuk mengunduh dataset.
3.  Jalankan semua sel secara berurutan.
