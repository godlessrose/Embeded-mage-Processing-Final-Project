import os
import glob


def convert_to_header():
    # Klasördeki .tflite uzantılı dosyaları bul
    tflite_files = glob.glob("*.tflite")

    if not tflite_files:
        print("❌ HATA: Klasörde hiç .tflite dosyası bulunamadı!")
        print("   Lütfen önce eğitim (training) kodunu çalıştırıp modelleri üret.")
        return

    # İlk bulunan dosyayı seç (Örn: SimpleSqueeze_quant.tflite)
    selected_model = tflite_files[0]
    header_path = "model.h"
    var_name = "mnist_model"  # Kodda kullanacağımız değişken adı

    print(f"✅ Bulunan Model: {selected_model}")
    print(f"⚙️  {header_path} dosyasına dönüştürülüyor...")

    with open(selected_model, 'rb') as f:
        data = f.read()

    with open(header_path, 'w') as f:
        f.write(f'#ifndef {var_name.upper()}_H\n')
        f.write(f'#define {var_name.upper()}_H\n\n')
        f.write(f'unsigned char {var_name}[] = {{\n')

        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n')

        f.write(f'}};\n\nunsigned int {var_name}_len = {len(data)};\n')
        f.write('#endif\n')

    print(f"🎉 BAŞARILI! '{header_path}' oluşturuldu.")
    print(f"👉 Şimdi bu 'model.h' dosyasını ESP32 projenin içine kopyala.")


if __name__ == "__main__":
    convert_to_header()
