# K Nearest Neighbor
- renk tanıma için makine öğrenmesi yaklaşımı olarak kullanılmaktadır.
- Girdi görüntüsündeki renk kanallarının histogramını çıkarır ve ortalama renk değerini elde edip dosyada depolar.
- Sınanacak görüntünün histogramındaki ortalama renk değerlerini bu dosyadaki değerlerle karşılaştırıp, en yakın olan K rengin sınıfı olarak yorumlar.

# pyhon modülleri açıklamaları

### KNN.py
- sınıflamalı makine öğrenimi gerçekleştirmek için(K nearest neighbor) yazılmış fonksiyonlar içeren betiktir.
- 3 kanallı bir görüntü bekler(numpy array olarak)

### colorDetectionCamera.py
- __camera.py__ dosyasının alt sınıfıdır. Renk tanıma için özelleştirilmiştir
- Makine modelinin eğitim sonuçlarını gerçek zamanlı ve görsel olarak yansıtabilmek için kamerayı kullanan bir betik.
- ekranda belirlenen alanı kapatacak biçimde renkli nesneyi koyun, model tahminleme yapıp yazacaktır

## camera.py
- Bu sınıf kameradan görüntü almak için kullanılır.
- Bu sınıfa ek olarak makine modeli verilebilir. Böylelikle modelin tahminlerini ekrana basar.
- makine modelinde öngörü gerçekleştirirken bu sınıfı kalıtan bir sınıf türetelim.
- Ekrana yansıtılacak görüntüyü önceden manipüle edebilelim.


### imageDataLoader.py
- Makine modeline verilecek görüntüleri ön işlemeden geçirmek için yazılmış betiktir.
- Bilgisayar belleğini etkili olarak kullanır, verilen yoldaki sınıfları bulur ve modele eğitim için aktarır
---