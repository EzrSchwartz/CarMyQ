#include <WiFi.h>
#include <HTTPClient.h>


bool connected = false;

// --- Wi-Fi credentials ---
const char* ssid = "SSID";
const char* password = "Password";

// --- Server endpoint ---

void setup() {
  Serial.begin(115200);

}


void loop() {
    if (!connected) {
        Serial.print("Connecting to Wi-Fi...");
        WiFi.begin(ssid, password);
        while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("Connected!");
    connected = true;
    }
    if (WiFi.status() != WL_CONNECTED) {
        connected = false;
    }
}