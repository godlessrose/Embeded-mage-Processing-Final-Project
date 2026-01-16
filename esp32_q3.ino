/*
 * EE 4065 Final Project - Question 2 Complete Solution
 * Hardware: ESP32-CAM (AI Thinker) + OV7670
 * Features:
 * - Brownout Protection Disabled (Prevents boot loops)
 * - PSRAM Usage (Prevents memory allocation errors)
 * - Color Inversion (Fixes black-text-on-white-paper issue)
 * - Web Server (Displays result on browser)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// --- KRİTİK: RESET SORUNUNU ÇÖZEN KÜTÜPHANELER ---
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// --- TFLITE KÜTÜPHANESİ (Chirale) ---
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Senin eğittiğin model dosyası
#include "model.h"

// ==========================================
// --- AYARLAR (BURAYI DOLDUR) ---
const char* ssid = "godlessrose";        // <-- Wifi Adın
const char* password = "UsRiot_191";  // <-- Wifi Şifren
// ==========================================

WebServer server(80);

// Hafıza Ayarı: PSRAM varsa 250KB ayırıyoruz, yoksa 60KB deniyoruz.
const int kTensorArenaSize = 250 * 1024; 
uint8_t *tensor_arena;

// Global Değişkenler (Sonuçları tutmak için)
int last_digit = -1;
float last_conf = 0.0;
unsigned long process_time = 0;

// KAMERA PIN AYARLARI (AI THINKER MODELİ İÇİN)
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

// TFLite Nesneleri
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;
tflite::MicroMutableOpResolver<12> tflOpsResolver; // Sayıyı artırdık garanti olsun

// --- WEB SAYFASI ---
void handleRoot() {
  String html = "<html><head>";
  // Sayfayı her 1 saniyede bir otomatik yenile
  html += "<meta http-equiv='refresh' content='1'>";
  html += "<style>body{font-family: Arial; text-align: center; margin-top: 50px;}";
  html += "h1{color: #444;} .big{font-size: 100px; font-weight: bold; color: #007BFF;}";
  html += "</style></head><body>";
  
  html += "<h1>ESP32 Rakam Tanimlama</h1>";
  
  if (last_digit != -1) {
    html += "<div class='big'>" + String(last_digit) + "</div>";
    html += "<h3>Guven Orani: %" + String(last_conf * 100) + "</h3>";
  } else {
    html += "<div class='big'>?</div>";
    html += "<h3>Analiz ediliyor veya sonuc belirsiz...</h3>";
  }
  
  html += "<p>Islem Suresi: " + String(process_time) + " ms</p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void setup() {
  // 1. BROWNOUT KAPAT (Reset atmasın)
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); 
  
  Serial.begin(115200);
  Serial.println("\n\n--- SISTEM BASLATILIYOR ---");

  // 2. BELLEK TAHSİSİ (PSRAM Kontrolü)
  if (psramFound()) {
    Serial.println("[OK] PSRAM Bulundu. Genis hafiza kullaniliyor.");
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  } else {
    Serial.println("[UYARI] PSRAM Yok! Standart RAM kullaniliyor (Riskli).");
    tensor_arena = (uint8_t*)malloc(60 * 1024);
  }

  if (tensor_arena == NULL) {
    Serial.println("[KRITIK HATA] Bellek ayrilamadi!");
    return;
  }

  // 3. KAMERA AYARLARI
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; 
  config.frame_size = FRAMESIZE_QQVGA; // 160x120      
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[HATA] Kamera baslatilamadi!");
    return;
  }
  Serial.println("[OK] Kamera Hazir.");
  
  // 4. WIFI BAĞLANTISI
  WiFi.begin(ssid, password);
  Serial.print("WiFi Baglaniyor: ");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\n[OK] Baglandi!");
  Serial.print("IP Adresi: http://");
  Serial.println(WiFi.localIP()); // <-- BU ADRESİ TARAYICIYA YAZACAKSIN

  server.on("/", handleRoot);
  server.begin();

  // 5. YAPAY ZEKA MODELİNİ YÜKLE
  tflModel = tflite::GetModel(model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[HATA] Model versiyonu uyumsuz!");
    return;
  }

  // Operatörleri Ekle (Hem Mean hem AvgPool ekledim, ne olur ne olmaz)
  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddAveragePool2D(); 
  tflOpsResolver.AddMean();          
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddQuantize();   
  tflOpsResolver.AddDequantize(); 

  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensor_arena, kTensorArenaSize);
  tflInterpreter = &static_interpreter;
  
  // Modeli Hafızaya Yerleştir
  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[HATA] AllocateTensors Failed! Hafiza yetmedi.");
    return;
  }

  inputTensor = tflInterpreter->input(0);
  outputTensor = tflInterpreter->output(0);
  Serial.println("[OK] Yapay Zeka Hazir. Test basliyor...");
}

void loop() {
  // Web sunucusunu dinle
  server.handleClient();
  
  // Görüntü Al
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) return;

  unsigned long start = millis();
  
  // --- GÖRÜNTÜ İŞLEME (RENK TERS ÇEVİRME) ---
  // Kağıt beyaz (255), mürekkep siyah (0).
  // Model siyah arka plan (0), beyaz yazı (255) istiyor.
  // Bu yüzden: Yeni_Pixel = 255 - Eski_Pixel
  
  if (inputTensor->type == kTfLiteInt8) {
      for (int i = 0; i < inputTensor->bytes; i++) {
         if (i < fb->len) {
            uint8_t pixel = fb->buf[i];
            
            // TERS ÇEVİRME İŞLEMİ
            uint8_t inverted_pixel = 255 - pixel;
            
            // Modele Yükle (INT8: -128 ile 127 arası)
            inputTensor->data.int8[i] = (int8_t)(inverted_pixel - 128);
         }
      }
  } 
  else { // Float32 kullananlar için yedek
      for (int i = 0; i < inputTensor->bytes; i++) {
          if (i < fb->len) {
             float pixel_norm = (float)(255 - fb->buf[i]) / 255.0f;
             inputTensor->data.f[i] = pixel_norm;
          }
      }
  }
  
  // Kamerayı serbest bırak
  esp_camera_fb_return(fb);

  // Tahmin Yürüt
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("HATA: Invoke basarisiz!");
    return;
  }
  process_time = millis() - start;

  // En yüksek skoru bul
  float max_score = 0;
  int best_digit = -1;
  
  Serial.print("Skorlar: [ ");
  for (int i = 0; i < 10; i++) {
    float score = 0;
    // Çıktı tipine göre oku
    if (outputTensor->type == kTfLiteInt8) {
         score = (float)(outputTensor->data.int8[i] - outputTensor->params.zero_point) * outputTensor->params.scale;
    } else {
        score = outputTensor->data.f[i];
    }
    
    // Serial.print(score); Serial.print(" "); // Detay istersen aç
    
    if (score > max_score) {
      max_score = score;
      best_digit = i;
    }
  }
  Serial.println("]");

  // Sonuçları Güncelle
  // Eşik değerini %40 yaptık ki en azından bir tahmin görelim
  if (max_score > 0.40) {
    last_digit = best_digit;
    last_conf = max_score;
    Serial.printf(">>> TESPIT: %d (Guven: %.2f)\n", best_digit, max_score);
  } else {
    // Çok düşükse de yaz ama web'e yansıtma veya soru işareti koy
    Serial.printf("Zayif Tahmin: %d (%.2f)\n", best_digit, max_score);
    last_digit = -1; 
  }

  // İşlemciyi biraz dinlendir (WiFi kopmaması için)
  delay(100);
}