#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "soc/soc.h" // Brownout sorunlarını engellemek için
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"

// ==========================================
// 1. WIFI AYARLARI (BURAYI DOLDUR)
// ==========================================
const char* ssid = "goddlessrose";
const char* password = "Us_Riot_191";

// ==========================================
// 2. PIN TANIMLARI (AI THINKER)
// ==========================================
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

// Soru 1 Hedefi
#define TARGET_PIXEL_COUNT 1000

// ==========================================
// 3. WEB SERVER VE GÖRÜNTÜ İŞLEME FONKSİYONU
// ==========================================
httpd_handle_t stream_httpd = NULL;

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  // Tarayıcıya bunun bir video akışı (MJPEG) olduğunu söyle
  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");
  if(res != ESP_OK) return res;

  while(true){
    // 1. Görüntüyü Yakala (Gri formatta)
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Kamera yakalama hatasi");
      res = ESP_FAIL;
    } else {
      
      // -----------------------------------------------------------
      // --- SORU 1: THRESHOLDING ALGORİTMASI BURADA BAŞLIYOR ---
      // -----------------------------------------------------------
      
      // A) Histogram Hesabı
      int histogram[256] = {0};
      for (size_t i = 0; i < fb->len; i++) {
        histogram[fb->buf[i]]++;
      }

      // B) Eşik Değeri Bulma (Size based extraction)
      int pixel_sum = 0;
      int optimal_threshold = 0;
      for (int i = 255; i >= 0; i--) {
        pixel_sum += histogram[i];
        if (pixel_sum >= TARGET_PIXEL_COUNT) {
          optimal_threshold = i;
          break;
        }
      }

      // C) Görüntüyü Değiştir (Siyah-Beyaz Yap)
      // Bu sayede tarayıcıda işlenmiş halini göreceksin
      for (size_t i = 0; i < fb->len; i++) {
        if (fb->buf[i] > optimal_threshold) {
          fb->buf[i] = 255; // Parlak nesne -> BEYAZ
        } else {
          fb->buf[i] = 0;   // Arka plan -> SİYAH
        }
      }
      // -----------------------------------------------------------
      // --- SORU 1 BİTİŞ ---
      // -----------------------------------------------------------

      // 2. Görüntüyü Tarayıcı İçin JPEG'e Çevir (Gri -> JPEG)
      // Kalite: 10-63 (Düşük sayı = yüksek kalite). Hız için 20-30 iyidir.
      bool jpeg_converted = fmt2jpg(fb->buf, fb->len, fb->width, fb->height, PIXFORMAT_GRAYSCALE, 30, &_jpg_buf, &_jpg_buf_len);
      
      esp_camera_fb_return(fb); // Ham görüntüyü bırak
      fb = NULL;

      if(!jpeg_converted){
        Serial.println("JPEG donusturme hatasi");
        res = ESP_FAIL;
      }
    }

    // 3. Görüntüyü Gönder (Chunk)
    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, "\r\n--frame\r\n", 12);
    }

    // JPEG Buffer'ını temizle
    if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    
    // Bağlantı koparsa döngüden çık
    if(res != ESP_OK){
      break;
    }
  }
  return res;
}

// Server Başlatma Fonksiyonu
void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri = {
    .uri       = "/",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
  }
}

// ==========================================
// 4. SETUP VE LOOP
// ==========================================
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Brownout dedektörünü kapat
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  
  // Kamera Ayarları
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
  config.pixel_format = PIXFORMAT_GRAYSCALE; // İŞLEME İÇİN GRİ FORMAT ŞART
  config.frame_size = FRAMESIZE_QVGA; // 320x240 (Hız için düşük tutuyoruz)
  config.jpeg_quality = 12;
  config.fb_count = 2; // Stream için buffer sayısı 2 olsun

  // Kamera Init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Kamera Hatasi: 0x%x", err);
    return;
  }

  // WiFi Bağlantısı
  WiFi.begin(ssid, password);
  Serial.print("WiFi Baglaniyor");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi Baglandi!");
  Serial.print("Kamera Yayini Surada: http://");
  Serial.println(WiFi.localIP());

  // Sunucuyu Başlat
  startCameraServer();
}

void loop() {
  // Loop boş kalabilir, çünkü server asenkron çalışıyor.
  delay(1000);
}