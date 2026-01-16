/*
 * ESP32-CAM MNIST OFFLINE MOD
 * 
 * Sonuçlar Serial Monitor'den (115200 baud) okunur.
 */
#include <Arduino.h>
#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- KÜTÜPHANELER ---
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h" // model.h dosyanın adı 'model_data' olmalı!

// --- BELLEK AYARLARI ---
// 
uint8_t *tensor_arena;
int kTensorArenaSize = 70 * 1024; 

// --- GLOBAL TFLITE NESNELERİ ---
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;
tflite::MicroMutableOpResolver<20> tflOpsResolver; 

// --- KAMERA PINLERİ (AI THINKER) ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// --------------------------------------------------------------------------
//  YAPAY ZEKA TAHMİN FONKSİYONU
// --------------------------------------------------------------------------
void run_inference() {
  // 1. Görüntüyü Al
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Kamera Hatasi!");
    return;
  }

  // 2. Görüntü İşleme (Preprocessing)
  // 160x120 -> 28x28 (Kırpma ve Küçültme)
  int w = fb->width;
  int h = fb->height;
  int start_x = (w - h) / 2; // Kare kırpma
  
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      // Pikseli bul
      int cam_x = start_x + (x * h / 28);
      int cam_y = (y * h / 28);
      int p_idx = cam_y * w + cam_x;
      if (p_idx >= fb->len) p_idx = fb->len - 1;
      
      uint8_t pixel = fb->buf[p_idx];
      
      // --- RENK AYARI (ÖNEMLİ) ---
      // Beyaz kağıda siyah kalemle yazıyorsan: AÇIK KALSIN (255 - pixel)
      // Siyah zemine beyaz tebeşir/ekran ise: SİL (sadece pixel kalsın)
      pixel = 255 - pixel; 
      
      // Threshold (Kontrast Artırıcı) - Rakamları netleştirir
      if (pixel < 80) pixel = 0;
      else if (pixel > 150) pixel = 255;

      int idx = (y * 28) + x;
      
      // Modele Yükle
      if (inputTensor->type == kTfLiteInt8) {
         inputTensor->data.int8[idx] = (int8_t)(pixel - 128);
      } else {
         inputTensor->data.f[idx] = (float)(pixel) / 255.0f;
      }
    }
  }

  // 3. Hafızayı Temizle (Kamerayla işimiz bitti, hemen serbest bırak)
  esp_camera_fb_return(fb);

  // 4. Modeli Çalıştır
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke Hatasi!");
    return;
  }

  // 5. Sonuçları Analiz Et
  float max_score = 0;
  int best_digit = -1;

  // Serial.print("Skorlar: ["); // Detay istersen aç
  for (int i = 0; i < 10; i++) {
    float score = 0;
    if (outputTensor->type == kTfLiteUInt8) {
        score = (float)(outputTensor->data.uint8[i] - outputTensor->params.zero_point) * outputTensor->params.scale;
    } else if (outputTensor->type == kTfLiteInt8) {
         score = (float)(outputTensor->data.int8[i] - outputTensor->params.zero_point) * outputTensor->params.scale;
    } else {
        score = outputTensor->data.f[i];
    }
    
    // Serial.printf("%.2f ", score); // Detay istersen aç
    if (score > max_score) {
      max_score = score;
      best_digit = i;
    }
  }
  // Serial.print("] ");

  // EŞİK DEĞERİ: %50'den emin değilse yazmasın (Karmaşayı önler)
  if (max_score > 0.5) {
    Serial.printf(">>> TAHMIN: %d (Guven: %% %.1f)\n", best_digit, max_score * 100);
  } else {
    // Emin değilse nokta koy (sistem çalışıyor mesajı)
    Serial.print("."); 
  }
}

// --------------------------------------------------------------------------
//  SETUP
// --------------------------------------------------------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  Serial.begin(115200);
  Serial.println("\n--- ESP32-CAM OFFLINE MNIST ---");

  // 1. PSRAM KONTROLÜ
  if (psramFound()) {
    Serial.println("[BILGI] PSRAM Aktif! (Rahat mod)");
    kTensorArenaSize = 1000 * 1024; // 1MB ayır (Kocaman yer var)
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  } else {
    Serial.println("[UYARI] PSRAM Yok! (Tasarruf modu)");
    // Web server olmadığı için dahili RAM buna yeter
    kTensorArenaSize = 70 * 1024; 
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  }
  
  if (tensor_arena == NULL) {
    Serial.println("[HATA] Hafiza ayrilamadi!");
    while(1);
  }

  // 2. KAMERA INIT
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; // Siyah Beyaz (Şart!)
  config.frame_size = FRAMESIZE_QQVGA;       // 160x120 (Hız için)
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[HATA] Kamera Baslatilamadi!");
    return;
  }
  Serial.println("[OK] Kamera Hazir.");

  // 3. MODEL YÜKLEME
  tflModel = tflite::GetModel(mnist_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[HATA] Model sema versiyonu uyumsuz!");
    while(1);
  }

  // Operatörleri Ekle (Hatayı önleyen tam liste)
  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddDepthwiseConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddAveragePool2D();
  tflOpsResolver.AddMean();             // <--- MEAN HATASI ÇÖZÜCÜ
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddAdd();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();
  tflOpsResolver.AddPad();
  
  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensor_arena, kTensorArenaSize);
  tflInterpreter = &static_interpreter;

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[HATA] Tensor Arena YETMEDI!");
    while(1);
  }

  inputTensor = tflInterpreter->input(0);
  outputTensor = tflInterpreter->output(0);
  Serial.println("[OK] Model Hazir! Rakam gosterin...");
}

// --------------------------------------------------------------------------
//  LOOP (Sürekli Tahmin)
// --------------------------------------------------------------------------
void loop() {
  run_inference();
  
  // İşlemciyi azıcık dinlendir (Aşırı ısınmayı önler)
  // Eğer çok hızlı aksın istersen delay'i 10 yapabilirsin.
  delay(200); 
}