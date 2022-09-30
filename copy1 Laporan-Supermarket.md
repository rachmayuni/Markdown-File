# Laporan Proyek 1 Machine Learning - Rachma Yuni Andari


# Domain Proyek
*Supermarket* adalah bagian kecil dari industri bisnis bersamaan dengan tujuan utamanya adalah menjual bahan kebutuhan pokok sehari-hari [1]. Pemberian layanan, ketersediaan produk adalah sesuatu yang penting untuk mempertahankan pelanggan atau mengundang pelanggan baru. Pelanggan pun tanpa sadar secara mudah mengakses dan membandingkan pelayanan dari berbagai supermarket yang pelanggan kunjungi. Kualitas pelayanan yang ditawarkan seharusnya memberikan kepuasan dan menciptakan loyalitas sehingga dapat meningkatkan **rating** pada *branch* toko terkait.


# Business Understanding
Model bisnis dalam proyek ini adalah distributor dan retail. Untuk memperoleh perbandingan rating yang baik, maka diterapkan automasi pada sistem untuk memprediksi perbandingan rating dari 3 supermarket dengan teknik *predictive modelling.*
Disediakan data mengenai pembelian yang dilakukan oleh pelanggan di **tiga supermarket (A, B, C)** dalam satu wilayah.
Maka tujuan/goals dari proyek ini adalah memprediksi **tingkat kepuasan pelanggan** berdasarkan rating. 

## Problem Statements
- Dari beberapa algoritma yang tersedia, model algoritma yang mana yang terbaik untuk kasus perkiraan estimasi rating supermarket?

## Goals
- Mengetahui algoritma yang terbaik pada kasus perkiraan estimasi rating supermarket

## Solutions Statements

Pada proyek ini digunakan 3 algoritma untuk mencapai pilihan solusi terbaik yang diinginkan,
yaitu:
- Lasso (L1 Regularization)
- KNN
- Random Forest

## Metodologi
Kita tahu bahwa rating merupakan variabel yang kontinu sehingga ini adalah permasalahan regresi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model regresi dengan rating sebagai target.

## Metrik
Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi.
Pengembangan model menggunakan 3 algoritma machine learning, yaitu L1 Regularization, KNN, dan Random Forest. Dari ketiga model, akan dipilih satu model yang punya nilai kesalahan prediksi terkecil.


# Data Understanding

Data yang digunakan dalam proyek ini adalah **Supermarket Sales Dataset** yang diunduh dari [Kaggle](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales). Pada proyek kali ini, tidak ada tambahan / penggabungan dari dataset file yang terpisah. Dataset ini cukup bersih dan tidak terlalu banyak memerlukan data cleaning.

Dataset ini memiliki 1000 jenis ID Invoice yang unik dan karakteristik berbagai macam kota dalam satu daerah. Karakteristik yang dimaksud adalah fitur non-numerik seperti: Invoice_ID, Branch, City, Costumer_type, Gender, Product_line, Date, Time, Payment_. Dan fitur numerik, seperti _Quantity, Unit_price, Tax_five_percent, Total, cogs, gmp, gi,_ dan _rating._

Ke tujuh belas fitur ini adalah fitur yang digunakan dalam menemukan pola data, sedangkan rating merupakan fitur target.

Masuk ke dalam proses pembacaan data (**data loading**).

Dataset yang digunakan adalah **supermarket_sales.csv**

Di sini kita mengunggah data ke Google Drive, lalu import library yang dibutuhkan.
Lalu simpan dataset di variable **stuff**

Untuk memudahkan, saya menghilangkan spasi dalam judul kolom dan menggantinya dengan _ (underscore) sebelum dataset diunggah ke Google drive. 

Output kode di atas memberikan informasi sebagai berikut:

- Ada 1000 baris (records) dalam dataset.

- Terdapat 17 kolom, yaitu: Invoice_ID, Branch, City, Costumer_type, Gender, Product_line, Unit_Price, Quantity, tax_five_percent Total, Date, Time, Payment, Cogs, Gross Margin Percentange, Gross Income, Rating.

## Exploratory Data Analysis - Deskripsi Variabel
Tahap ini adalah tahap menganalisis karakteristik, menemukan pola, anomaly, dan memeriksa asumsi data.

### Jenis Variabel dalam Dataset
-   **Invoice ID**: merupakan kode slip invoice unik yang dihasilkan oleh computer (Unique)
-   **Branch**: merupakan cabang dari supercenter (3 cabang tersedia diidentifikasi oleh A, B, dan C)
-   **City**: merupakan lokasi/kota dari supercenters
-   **Customer_type**: merupakan tipe dari customers, dicatat sebagai Anggota / Normal (tergantung ada atau tidaknya kartu keanggotaan)
-   **Gender**: merupakan jenis kelamin dari customer
-   **Product_line**: merupakan grup kategorisasi barang umum; Elektronik, Mode, Makanan dan minuman, Kesehatan dan Kecantikan, Rumah dan Gaya Hidup, Olahraga.
-   **Unit_Price**: merupakan harga setiap produk dalam satuan dolar ($)
-   **Quantity**: merupakan jumlah produk yang dibeli oleh customer
-   **Tax**: merupakan 5% biaya pajak untuk setiap pembelian dari customer
-   **Total**: merupakan total harga termasuk pajak
-   **Date**: merupakan tanggal pembelian dari Januari 2019 - Maret 2019
-   **Time**: merupakan waktu pembelian 10 pagi - 9 malam
-   **Payment**: merupakan pembayaran yang digunakan oleh customer (Cash, Credit card, atau e-wallet)
-   **COGS**: merupakan Harga pokok penjualan
-   **Gross margin percentage (gmp)**: merupakan persentase margin kotor
-   **Gross income (gi)**: merupakan pendapatan kotor
-   **Rating**: merupakan tingkat kepuasan pelanggan (skala 1 - 10)

### Fitur Tidak Berguna (Redundant)

Kolom **Invoice_ID** tidak terlalu berguna karena kolom tersebut unik dan tidak terlalu berpengaruh terhadap data-data baru.

Yang saat ini berpengaruh adalah City dan Branch. Disamping mereka berkaitan, di sana mereka menjadi bagian terpenting untuk menentukan dan sebagai prediksi perbandingan kepuasan pelanggan.

Selanjutnya adalah mengecek informasi pada dataset dengan fungsi info()
*<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 17 columns):*
| # | Column | Non-Null Count | Dtype |
|---|--------|----------------|-------|
|  0 |    Invoice_ID    |    1000 non-null            |    object   | 
| 1  |    Branch    |          1000 non-null      |    object   |
|  2 |     City   |         1000 non-null       |    object   | 
|  3 |    Customer_type    |        1000 non-null        |     object  | 
|  4 |   Gender     |         1000 non-null       |    object   | 
| 5  |  Product_line      |       1000 non-null         |     float64  |
|  6 |    Unit_price    |       1000 non-null         |     int64  | 
|  7 |     Quantity   |        1000 non-null        |    float64   | 
|  8 |   Tax_five_percent     |        1000 non-null        |    float64   | 
| 9|    Total    |       1000 non-null         |      object |
|  10 |   Date     |      1000 non-null          |     object  | 
|  11 |     Time   |     1000 non-null           |      object | 
|  12 |   Payment     |    1000 non-null            |     float64   | 
| 13 |    cogs    |        1000 non-null        |   float64     |
|  14 |     gmp   |          1000 non-null      |    float64    | 
|  15 |    gi    |      1000 non-null           |       float64 |
|  16 |     Rating   |      1000 non-null           |       float64 |
*dtypes: float64(7), int64(1), object(9) memory usage: 132.9+ KB*

Dari output terlihat bahwa:
- Terdapat 9 kolom kategori dengan tipe object
- Terdapat 8 kolom numerik dengan tipe int dan float


## Exploratory Data Analysis - Cek Missing Value
Hasil dari penggunaan fungsi describe(), kolom gmp atau gross margin percentage adalah 0.
Mari kita cek missing value pada kolom gmp.

Ternyata 0.0000 adalah representasi dari standar deviasi dan bukan merupakan nilai min dan max dalam datanya.

Selanjutnya adalah menangani outliers.

### Menangani Outliers
Pada proyek ini, deteksi outliers akan dilakukan dengan teknik visualisasi data (boxplot).
Boxplot menunjukkan distribusi data kuantitatif dengan cara yang memfasilitasi perbandingan antar variabel atau lintas tingkat variabel kategori. Kotak menunjukkan kuartil dari kumpulan data sementara garis di luarnya untuk menunjukkan sisa distribusi [2].

1. Unit_price
![gambar 7](https://github.com/rachmayuni/Foto-Proyek-Prediksi-1---Dicoding/blob/main/tujuh.png?raw=true)
Dapat terlihat pada gambar di atas bahwa outlier di kolom unit_price tidak ada. Minimum terletak di angka < 20 dan maximum terletak di angka 100.

2. Total
![gambar 8](https://github.com/rachmayuni/Foto-Proyek-Prediksi-1---Dicoding/blob/main/delapan.png?raw=true)
Pada gambar di atas, IQR (range antara Q1 dan Q3) terletak di antara angka > 100 dan angka < 500. Dan terlihat bahwa terdapat beberapa outlier di sana.

3. Tax_five_percent
Begitu juga dengan boxplot tax_five_percent, bahwa ada beberapa outlier di sebelah kanan, sebagai representasi ada data yang melebihi maximum(Q3 + 1.5*IQR)


Dataset sekarang telah bersih dan memiliki 991 sampel dan memiliki 17 kolom.

## Exploratory Data Analysis - Deskripsi Variabel
Pertama, bagi fitur dataset menjadi empat bagian, yaitu: time_features, date_features, binary_features, dan nominal_features.

Kolom Date akan diatasi secara terpisah karena pada awalnya mereka adalah categorical, lalu harus diubah menjadi informasi numerical di dalamnya. Pull out hari, pull out bulan, dan pull out tahun.
Begitu juga dengan time. Kolom akan diatasi dengan pull out menjadi menit dan jam pada kolom terpisah.
Pada kolom costumer_type dan gender, mereka hanya memiliki dua pilihan, yaitu: 0 dan 1. sehingga termasuk pada fitur binary.
Kolom sisanya (Branch, City, Product_line, payment) adalah nominal. 

**Binary_features**
![gambar 12](https://github.com/rachmayuni/Foto-Proyek-Prediksi-1---Dicoding/blob/main/12.png?raw=true)
Presentase pelanggan yang member dan pelanggan yang bukan member (normal) hampir sepadan.

**Nominal_features**
![gambar 13](https://github.com/rachmayuni/Foto-Proyek-Prediksi-1---Dicoding/blob/main/13.png?raw=true)

Untuk fitur numerik / nominal, dapat dilihat melalui histogram berikut.
![gambar 14](https://github.com/rachmayuni/Foto-Proyek-Prediksi-1---Dicoding/blob/main/14.png?raw=true)

Histogram untuk variabel **“rating”** yang merupakan fitur target (label) pada data, dapat diperolah informasi sebagai berikut:
Rating stabil naik turun seiring (stabil) dengan bertambahnya jumlah sampel.


# Data Preparation

Pada bagian ini, akan dilakukan empat tahap persiapan data, yaitu:
- Encoding Fitur Date and Time
- Pembagian dataset dengan fungsi
- Standarisasi

**Encoding Fitur**

Date and Time
Melakukan transformasi data (pull out), maka harus diimplementasikan fungsi fit dan transform.
Pertama adalah buat date encoder.
Fit menyimpan informasi tentang data sebelum transformasi dilakukan. Sehingga untuk fungsi fit, bisa langsung dikembalikan saja pada nilai X (return).
Mulailah eksekusi pull out pada fungsi transform, karena pada data frame sebelumnya, kolom date masih menjadi kesatuan yang hanya dipisahkan oleh /.
Sehingga disini kita menggunakan fungsi lambda untuk mengeluarkan elemen-elemen.
> pd.to_datetime(X_train['Time']).apply(lambda x: x.minute)

Variabel date, month, year, hour bisa kita aplikasikan juga pada fungsi tersebut.
Sehingga hasil akhir pada tahap pull out ini adalah elemen bisa terpisah di kolom yang berbeda menjadi tipe integer.

Selanjutnya, pada fitur selain Date dan Time,
maka digunakan pipeline.
Masing-masing pipeline disimpan dalam variabelnya masing-masing.
Transformer digunakan untuk tipe informasi yang berbeda.

Pada binary, digunakan OrdinalEncoder()
Pada date, digunakan DateEncoder()
Pada time, digunakan TimeEncoder()
Pada onehot, digunakan OneHotEncoder()

**Train Test Split**

Membagi dataset menjadi **data latih (train)** dan **data uji (test)** dilakukan sebelum model dibuat. Ada beberapa data yang perlu dipertahankan untuk menguji seberapa baik generalisasi model terhadap data baru. Yang kita tahu bahwa data uji (test set) adalah bisa dibilang ‘data baru’, maka perlu dilakukan proses transformasi dalam data latih.

Tujuan membagi dataset menjadi data uji dan data latih adalah supaya data uji tidak kotor.

Pada **train_test_split** pada proyek ini, proporsi pembagian data latih dan data test adalah **70:30**. Kemudian digunakan juga fungsi train_test_split dari library sklearn.

- Total # of sample in whole dataset: 1000
- Total # of sample in train dataset: 700
- Total # of sample in test dataset: 300

**Standarisasi**
Fitur standarisasi hanya diterapkan di fitur data latih. Kapan kita menerapkan nya pada data test? Nanti pada tahap evaluasi.

Selanjutnya adalah, mengkombinasikan semuanya dan memberitahukan pipeline; kolom mana yang menjadi target dari setiap transformer. Lalu menyimpannya pada variabel preprocessor.

Saatnya melakukan pengubahan skala menjadi sama.
Pertama, digunakan **StandardScaler** sehingga pada kolom, mereka mempunyai mean 0 dan variasi 1.

Setelah itu, membangun final pipeline dengan mendefinisikan step dari beberapa langkah yang telah dilakukan di atas yaitu preprocessor, scaler, dan regressor.
Kemudian fit pipeline, dengan mengeksekusi data latih.


## Modelling
Dalam proyek ini digunakan tiga algoritma untuk mengembangkan model machine learning. Kemudian akan dievaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik.
1. **Linear Regression (L1 Regularization)**
	Algoritma L1 Regularization merupakan model yang menyesuaikan bobot sesuai dengan nilai absolut dari bobot.
	Model ini menggerakkan bobot fitur yang tidak relevan menjadi tepat 0 dan menghapus fitur model tersebut dari model. Model regresi yang menggunakan teknik regularisasi L1 disebut dengan Regresi Lasso.
	- **Kelebihan L1 Regularization**
LASSO adalah teknik regularisasi model untuk menghindari overfitting, dapat membantu untuk mengimplementasikan fitur seleksi otomatis.
	- **Kekurangan L1 Regularization**
Menyetel alpha terlalu rendah dapat menghilangkan efek regulasi dan dapat menyebabkan underfitting

2. **K-Nearest Neighbors**
Algoritma KNN menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. Setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.
	- **Kelebihan KNN**
Efektif kalau data training nya besar, KNN sangat mudah diimplementasikan. Hanya ada dua parameter yang diperlukan untuk mengimplementasikan KNN yaitu nilai K dan fungsi jarak (misalnya Euclidean atau Manhattan dll.)
	- **Kekurangan KNN**
Tidak bekerja bagus dengan dataset yang besar dan dimensi tinggi, butuh feature scaling, terlalu sensitif dengan data yang noisy, missing values, and outliers.

3. **Random Forest**
Algoritma Random Forest merupakan model yang termasuk ke dalam ensemble learning. Di mana ensemble learning adalah model prediksi dari beberapa model dan bekerja secara bersama-sama sehingga tingkat keberhasilan akan lebih tinggi disbanding model yang bekerja sendiri.
	- **Kelebihan Random Forest**
Runtime cukup cepat sehingga dapat menangani *imbalance data*
	- **Kekurangan Random Forest**
Ketika digunakan untuk regresi, model tidak dapat memprediksi di luar jangkauan dalam data training dan kemungkinan dataset akan over-fit dan noisy.


## Evaluation

Selisih nilai sebenarnya dengan nilai prediksi disebut **error.**
Metrik yang mengukur seberapa kacil error tersebut.
Metrik yang digunakan pada prediksi ini adalah R Square. Tidak digunakan MSE karena berdasarkan pemahaman data di awal, outlier tidak terlalu banyak. R2 digunakan untuk mengevaluasi kinerja model regresi linier. Ini adalah jumlah variasi dalam atribut dependen output yang dapat diprediksi dari variabel independen input. Ini digunakan untuk memeriksa seberapa baik hasil yang diamati oleh model, tergantung pada rasio total deviasi hasil yang dijelaskan oleh model.


$$ R^2 = {1 - {SSres} \over SStot} $$


- Linear Regression (L1 Regularization)		R^2 Score: -0.00064
- K-Nearest Neighbors 									R^2 Score: -0.13380
- Random Forest											R^2 Score: -0.11337

Model selesai dilatih dengan 3 algoritma, yaitu L1 Regularization, KNN, dan Random Forest.


## Conclusion
Dapat dilihat bagaimana model algoritma yang berbeda telah dibandingkan dan dapat disimpulkan bahwa model yang paling baik, yaitu model yang menggunakan algoritma **L1 Regularization. (Lasso)**. Walau hanya linear regresi tapi dia memiliki skor R^2 tertinggi.


## References
[1] T. Utomo, "PERSAINGAN BISNIS RITEL: TRADISIONAL VS MODERN", _Ejournal.stiepena.ac.id_, 2022. [Online]. Available: http://ejournal.stiepena.ac.id/index.php/fe/article/view/151. [Accessed: 29- Sep- 2022]
[2] "Understanding Boxplots", _Built In_, 2022. [Online]. Available: https://builtin.com/data-science/boxplot. [Accessed: 29- Sep- 2022]

