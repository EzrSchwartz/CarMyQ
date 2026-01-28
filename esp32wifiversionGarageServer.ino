#include <WiFi.h>
#include <HTTPClient.h>

#define SENSOR_PIN 1  // cadence sensor input
#define INPUT_PIN 4  // Pin receiving data
#define OUTPUT_PIN 3  // Pin sending data out

bool commandConsumed = false;

// --- Wi-Fi credentials ---
const char* ssid = "SSID";
const char* password = "Password";
bool doorOpen = false;

// --- Server endpoint ---

bool lastState = HIGH;

void setup() {
  pinMode(SENSOR_PIN, INPUT_PULLUP);
  pinMode(INPUT_PIN, INPUT);
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, OUTPUT_PIN);
  Serial.begin(115200);
  WiFi.mode(WIFI_AP);

  bool result = WiFi.softAP(ssid,password); 
  if (result) {
    Serial.println("WiFi AP enabled successfully!");
  } else {
    Serial.println("WiFi AP failed to enable.");
  }
  
  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(myIP);
}


void loop() {
  bool sensorState = digitalRead(SENSOR_PIN); 

  if (sensorState != lastState) {  
    if (sensorState == 1) {
      Serial.println("Door CLOSED (magnet away)");
    } else {
      Serial.println("Door OPEN (magnet close)");
    }
    lastState = sensorState;
  }
  checkCommand();
  delay(5); 

  if WiFi.softAPgetStationNum() > 0 && doorOpen == false{
    Serial.println("Client connected to AP");
    relayPins();
  } else {
    Serial.println("No clients connected to AP");
  }

}


void relayPins() {
  unsigned long startTime = millis();
  
  while (millis() - startTime < 3000) {
    digitalWrite(OUTPUT_PIN, digitalRead(INPUT_PIN));
  }
  startTime = millis();

  while (millis() - startTime < 10000) {
    digitalWrite(OUTPUT_PIN, OUTPUT_PIN);
}
}
