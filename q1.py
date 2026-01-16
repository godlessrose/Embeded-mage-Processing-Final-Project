import cv2
import numpy as np
import os

# Dosya adı
image_path = 'test.jpg'


def main():
    # 1. Dosya Kontrolü (Hata ayıklama için çok önemli)
    if not os.path.exists(image_path):
        print(f"HATA: '{image_path}' dosyası bulunamadı!")
        print(f"Python şurada arıyor: {os.getcwd()}")
        print("Lütfen resmi bu python dosyasıyla aynı klasöre koyduğundan emin ol.")
        return

    # 2. Resmi Gri Tonlamalı Oku
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("HATA: Dosya var ama okunamadı (bozuk olabilir).")
        return

    # Resim boyutlarını yazdıralım (Doğru yüklendi mi emin olalım)
    height, width = img.shape
    print(f"Resim Yüklendi: {width}x{height} piksel")

    # 3. Histogram Analizi (Piksel sayma işlemi)
    # 0-255 arasındaki her parlaklık değerinden kaç adet var?
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    target_pixels = 1000  # Soruda istenen hedef [cite: 11]
    count = 0
    optimal_threshold = 0

    # 4. En parlaktan (255) en koyuya (0) doğru pikselleri topla
    print("Hesaplanıyor...")
    for i in range(255, -1, -1):
        count += hist[i][0]  # O parlaklıktaki piksel sayısını ekle

        # Eğer toplam piksel sayısı 1000'i geçerse dur
        if count >= target_pixels:
            optimal_threshold = i
            break

    # 5. Sonuçları Yazdır
    print("-" * 30)
    print(f"HEDEF PIKSEL: {target_pixels}")
    print(f"BULUNAN ESIK DEGERI (THRESHOLD): {optimal_threshold}")
    print(f"SECILEN TOPLAM PIKSEL: {int(count)}")
    print("-" * 30)

    # 6. Görüntüleme (Sağlama yapma)
    # Bulunan eşik değerine göre resmi siyah-beyaz yap
    _, thresh_img = cv2.threshold(img, optimal_threshold, 255, cv2.THRESH_BINARY)

    cv2.imshow("Orijinal Resim", img)
    cv2.imshow("Sonuc (Sadece En Parlak 1000 Piksel)", thresh_img)

    print("Pencereleri kapatmak için herhangi bir tuşa bas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()