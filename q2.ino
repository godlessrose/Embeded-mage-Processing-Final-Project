#include <Arduino.h>
#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model dosyanız (Adı projene göre değişebilir)
#include "my_first_project_model.h"

// --- AYARLAR ---
// Gürültü engelleme eşiği. Görüntü işlendikten sonra 80'in altındaki her şey simsiyah yapılır.
// Eğer silik yazıyorsan bu değeri düşür (örn: 50).
#define NOISE_THRESHOLD 80 

// Bellek
const int kTensorArenaSize = 86 * 1024; 
uint8_t *tensor_arena;

// TFLite
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;
tflite::MicroMutableOpResolver<25> tflOpsResolver; 

// Pin Tanımları (AI-Thinker ESP32-CAM)
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

void run_inference() {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Kamera Hatasi!");
    return;
  }

  int model_w = inputTensor->dims->data[2];
  int model_h = inputTensor->dims->data[1];
  
  // Görüntüden kırpılacak kare alanın hesabı
  int crop_size = (fb->width < fb->height) ? fb->width : fb->height;
  int start_x = (fb->width - crop_size) / 2;
  int start_y = (fb->height - crop_size) / 2;

  // --- ADIM 1: ANALİZ (Min/Max Bulma) ---
  // Görüntüdeki en parlak (kağıt) ve en karanlık (kalem) noktayı buluyoruz.
  // Bu sayede ortam ışığı ne olursa olsun kontrastı ayarlayabiliriz.
  uint8_t min_val = 255;
  uint8_t max_val = 0;
  
  // Performans için her pikseli değil, örnekleme yaparak tarayabiliriz ama
  // model küçük olduğu için tam tarama yapalım.
  for (int y = 0; y < model_h; y++) {
    for (int x = 0; x < model_w; x++) {
      int cam_x = start_x + (x * crop_size / model_w);
      int cam_y = start_y + (y * crop_size / model_h);
      int p_idx = cam_y * fb->width + cam_x;
      uint8_t pixel = fb->buf[p_idx];
      
      if (pixel < min_val) min_val = pixel;
      if (pixel > max_val) max_val = pixel;
    }
  }

  // Eğer görüntü tamamen tek renkse (örn: kamera kapalı veya boş kağıt) işlem yapma
  if ((max_val - min_val) < 20) {
    esp_camera_fb_return(fb);
    // Serial.println("Goruntu cok duz (Kontrast yok). Pas geciliyor.");
    return; 
  }

  // --- ADIM 2: İŞLEME VE MODELE AKTARMA ---
  for (int y = 0; y < model_h; y++) {
    for (int x = 0; x < model_w; x++) {
      
      // 1. Pikseli kameradan al
      int cam_x = start_x + (x * crop_size / model_w);
      int cam_y = start_y + (y * crop_size / model_h);
      int p_idx = cam_y * fb->width + cam_x;
      uint8_t pixel = fb->buf[p_idx];

      // 2. Dinamik Kontrast Germe (En önemli kısım burası!)
      // Piksel değerini min ve max aralığından 0-255 aralığına genişletiyoruz.
      float normalized = (float)(pixel - min_val) / (max_val - min_val); // 0.0 ile 1.0 arası
      int processed_pixel = normalized * 255;

      // 3. Renk Tersleme (Beyaz Kağıt -> Siyah Arka Plan)
      processed_pixel = 255 - processed_pixel;

      // 4. Gürültü Temizleme (Thresholding)
      // Ters çevirdikten sonra arka plan (kağıt dokusu) gri kalabilir.
      // Eğer piksel değeri eşiğin altındaysa (koyu griyse) tam siyah yap.
      if (processed_pixel < NOISE_THRESHOLD) {
        processed_pixel = 0;
      } else {
        // Kalan kısmı parlat (Yazıyı daha belirgin yap)
        processed_pixel = map(processed_pixel, NOISE_THRESHOLD, 255, 0, 255);
      }

      // 5. Tensöre Yaz
      int idx = y * model_w + x;
      if (inputTensor->type == kTfLiteInt8) {
        // Int8 Quantized Model (-128 ile 127 arası)
        inputTensor->data.int8[idx] = (int8_t)(processed_pixel - 128);
      } else {
        // Float Model (0.0 ile 1.0 arası)
        inputTensor->data.f[idx] = (float)processed_pixel / 255.0f;
      }
    }
  }

  esp_camera_fb_return(fb); 

  // --- RUN ---
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke Hatasi!");
    return;
  }

  // --- ÇIKTI ANALİZİ ---
  int num_classes = outputTensor->dims->data[1];
  if (num_classes == 0) num_classes = outputTensor->dims->data[0];

  float max_score = -100;
  int best_digit = -1;

  Serial.print("Skorlar: ");
  for (int i = 0; i < num_classes; i++) {
    float score = 0;
    if (outputTensor->type == kTfLiteUInt8) {
      score = (outputTensor->data.uint8[i] - outputTensor->params.zero_point) * outputTensor->params.scale;
    } else if (outputTensor->type == kTfLiteInt8) {
      score = (outputTensor->data.int8[i] - outputTensor->params.zero_point) * outputTensor->params.scale;
    } else {
      score = outputTensor->data.f[i];
    }
    
    // Sadece yüksek ihtimalleri yazdıralım ki ekran kirlenmesin
    if (score > 0.01) {
       Serial.printf("[%d]:%%%.1f  ", i, score * 100);
    }

    if (score > max_score) {
      max_score = score;
      best_digit = i;
    }
  }
  Serial.println();

  // Güven Eşiği (Threshold): %60'ın altındaysa emin değilim de.
  if (max_score > 0.60) {
    Serial.printf(">>> TAHMIN: %d (Guven: %% %.1f)\n", best_digit, max_score * 100);
    Serial.println("--------------------------------");
  } else {
    Serial.println(">>> TAHMIN: ? (Emin degilim)");
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  Serial.println("--- SYSTEM START ---");

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
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_QQVGA; 
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Kamera baslatilamadi!");
    while(1);
  }

  tflModel = tflite::GetModel(model_data);
  
  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddDepthwiseConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddAveragePool2D(); // Bazı modellerde gerekir
  tflOpsResolver.AddMean();
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddAdd();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();
  tflOpsResolver.AddLogistic();
  tflOpsResolver.AddPad();

  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensor_arena, kTensorArenaSize);
  tflInterpreter = &static_interpreter;

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors Hatasi!");
    while(1);
  }

  inputTensor = tflInterpreter->input(0);
  outputTensor = tflInterpreter->output(0);
  
  Serial.println("Sistem Hazir.");
}

void loop() {
  run_inference();
  delay(500); 
}