#include <WiFi.h>
#include <HTTPClient.h>

#define SENSOR_PIN 1  // cadence sensor input
#define INPUT_PIN 4  // Pin receiving data
#define OUTPUT_PIN 3  // Pin sending data out

bool commandConsumed = false;

// --- Wi-Fi credentials ---
const char* ssid = "SSID";
const char* password = "Password";

// --- Server endpoint ---
const char* serverUrl = "http://ServerPortaddress"; // replace with your server

bool lastState = HIGH; // start with magnet away (door open)

void setup() {
  pinMode(SENSOR_PIN, INPUT_PULLUP);
  pinMode(INPUT_PIN, INPUT);
  pinMode(OUTPUT_PIN, OUTPUT);
  digitalWrite(OUTPUT_PIN, OUTPUT_PIN);
  Serial.begin(115200);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected!");
}

void loop() {
  bool sensorState = digitalRead(SENSOR_PIN);  // 0 = magnet near (closed), 1 = magnet away (open)

  if (sensorState != lastState) {  // only act on state change
    if (sensorState == 0) {
      Serial.println("Door CLOSED (magnet present)");
      sendUpdate("closed");
    } else {
      Serial.println("Door OPEN (magnet away)");
      sendUpdate("open");
    }
    lastState = sensorState;
  }
  checkCommand();
  delay(200);  // debounce / readable output
}

void checkCommand() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(String(serverUrl) + "/command");
    int code = http.GET();

    if (code == 200) {
      String response = http.getString();
      response.trim();  // IMPORTANT: removes \n \r
      Serial.println(response);
      if (response == "ACTIVATE") {
        commandConsumed = true;
        relayPins();
      }

      if (response == "NONE") {
        commandConsumed = false;  // re-arm trigger
      }
    }
    http.end();
  }
}

void sendUpdate(const char* state) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"door_state\":\"";
    payload += state;
    payload += "\"}";

    int httpResponseCode = http.POST(payload);
    if (httpResponseCode > 0) {
      Serial.printf("Update sent, response: %d\n", httpResponseCode);
    } else {
      Serial.printf("Error sending update: %s\n", http.errorToString(httpResponseCode).c_str());
    }
    http.end();
  } else {
    Serial.println("Wi-Fi not connected, cannot send update");
  }
}



void relayPins() {
  unsigned long startTime = millis();
  
  // Relay data for 3 seconds
  while (millis() - startTime < 3000) {
    // Read the input pin and write it to the output pin
    digitalWrite(OUTPUT_PIN, digitalRead(INPUT_PIN));
  }
  startTime = millis();

  while (millis() - startTime < 10000) {
    digitalWrite(OUTPUT_PIN, OUTPUT_PIN);
}
}





