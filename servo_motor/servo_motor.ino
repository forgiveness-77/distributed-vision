#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

const char* ssid = "RCA-OUTDOOR";
const char* password = "RCA@2025";

const char* mqtt_server = "157.173.101.159";  
const int mqtt_port = 1883;

const char* topic_servo_angle = "vision/n1ghtc0d3/movement";

const int SERVO_PIN = D1;  

WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;

int currentAngle = 90;  

void setup() {
  Serial.begin(9600);
  
  Serial.println("\n===== Simple Servo Controller =====");

  myServo.attach(SERVO_PIN);
  myServo.write(currentAngle);
  delay(1000);
  
  connectToWiFi();
  
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  
  Serial.println("Ready to receive angles on topic: face_tracking/servo_angle");
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {

  char message[length + 1];
  memcpy(message, payload, length);
  message[length] = '\0';

  if (String(topic) == topic_servo_angle) {
    int newAngle = atoi(message);
    

    if (newAngle >= 0 && newAngle <= 180) {
      currentAngle = newAngle;
      myServo.write(currentAngle);
      
      Serial.print("Servo moved to: ");
      Serial.print(currentAngle);
      Serial.println(" degrees");
    }
  }
}

void reconnect() {

  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    
    if (client.connect("servo_controller")) {
      Serial.println("connected!");
      client.subscribe(topic_servo_angle);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds...");
      delay(5000);
    }
  }
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}