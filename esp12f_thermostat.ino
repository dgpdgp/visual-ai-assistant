#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <WiFiManager.h>
#include <LittleFS.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// --- NRF24 Custom SoftSPI Class ---
class NRF24 {
private:
    uint8_t pin_sck, pin_mosi, pin_miso, pin_ce, pin_csn;

    uint8_t spiByte(uint8_t b) {
        uint8_t ret = 0;
        for (uint8_t i = 0; i < 8; i++) {
            if (b & (1 << (7 - i))) digitalWrite(pin_mosi, HIGH);
            else digitalWrite(pin_mosi, LOW);
            digitalWrite(pin_sck, HIGH);
            ret <<= 1;
            if (digitalRead(pin_miso)) ret |= 1;
            digitalWrite(pin_sck, LOW);
        }
        return ret;
    }

    void writeRegister(uint8_t reg, uint8_t value) {
        digitalWrite(pin_csn, LOW);
        spiByte(0x20 | reg); // W_REGISTER
        spiByte(value);
        digitalWrite(pin_csn, HIGH);
    }

    void writeRegister(uint8_t reg, const uint8_t* buf, uint8_t len) {
        digitalWrite(pin_csn, LOW);
        spiByte(0x20 | reg); // W_REGISTER
        for (uint8_t i = 0; i < len; i++) spiByte(buf[i]);
        digitalWrite(pin_csn, HIGH);
    }

    uint8_t readRegister(uint8_t reg) {
        digitalWrite(pin_csn, LOW);
        spiByte(reg); // R_REGISTER
        uint8_t val = spiByte(0xFF);
        digitalWrite(pin_csn, HIGH);
        return val;
    }

public:
    NRF24(uint8_t sck, uint8_t mosi, uint8_t miso, uint8_t ce, uint8_t csn) :
        pin_sck(sck), pin_mosi(mosi), pin_miso(miso), pin_ce(ce), pin_csn(csn) {}

    void begin() {
        pinMode(pin_sck, OUTPUT);
        pinMode(pin_mosi, OUTPUT);
        pinMode(pin_miso, INPUT);
        pinMode(pin_ce, OUTPUT);
        pinMode(pin_csn, OUTPUT);

        digitalWrite(pin_sck, LOW);
        digitalWrite(pin_ce, LOW);
        digitalWrite(pin_csn, HIGH);

        delay(100);
        writeRegister(0x00, 0x0E); // CONFIG: PWR_UP=1, PRIM_RX=0, EN_CRC=1, CRCO=1 (0x0E)
        writeRegister(0x01, 0x01); // EN_AA: Enable auto ack on pipe 0
        writeRegister(0x02, 0x01); // EN_RXADDR: Enable data pipe 0
        writeRegister(0x03, 0x03); // SETUP_AW: 5 bytes address
        writeRegister(0x04, 0x2F); // SETUP_RETR: 15 retries, 750us delay
        writeRegister(0x05, 76);   // RF_CH: Channel 76
        writeRegister(0x06, 0x27); // RF_SETUP: 250kbps, 0dBm (MAX)

        uint8_t addr[5] = {'T','H','R','M','1'};
        writeRegister(0x0A, addr, 5); // RX_ADDR_P0
        writeRegister(0x10, addr, 5); // TX_ADDR
        writeRegister(0x11, 32);      // RX_PW_P0: 32 bytes
    }

    bool send(void* buf, uint8_t len) {
        writeRegister(0x07, 0x7E); // Clear interrupts

        digitalWrite(pin_csn, LOW);
        spiByte(0xE1); // FLUSH_TX
        digitalWrite(pin_csn, HIGH);

        digitalWrite(pin_csn, LOW);
        spiByte(0xA0); // W_TX_PAYLOAD
        uint8_t* p = (uint8_t*)buf;
        for (uint8_t i = 0; i < len; i++) spiByte(p[i]);
        digitalWrite(pin_csn, HIGH);

        digitalWrite(pin_ce, HIGH);
        delayMicroseconds(15);
        digitalWrite(pin_ce, LOW);

        uint32_t start = millis();
        uint8_t status = 0;
        while (millis() - start < 100) {
            status = readRegister(0x07);
            if (status & (1 << 5)) { // TX_DS
                writeRegister(0x07, (1 << 5)); // clear TX_DS
                return true;
            }
            if (status & (1 << 4)) { // MAX_RT
                writeRegister(0x07, (1 << 4)); // clear MAX_RT
                return false;
            }
            delay(1);
        }
        return false;
    }
};

// --- Config ---
#define OLED_SDA 14
#define OLED_SCL 12
#define DS18B20_PIN 16
#define BUTTON_PIN 0

#define NRF_SCK 13
#define NRF_MOSI 5
#define NRF_MISO 4
#define NRF_CE 2
#define NRF_CSN 15

Adafruit_SSD1306 display(128, 64, &Wire, -1);
OneWire oneWire(DS18B20_PIN);
DallasTemperature sensors(&oneWire);
NRF24 radio(NRF_SCK, NRF_MOSI, NRF_MISO, NRF_CE, NRF_CSN);
ESP8266WebServer server(80);

struct Packet {
    uint8_t cmd;
    int16_t temp;
    int16_t target;
    uint8_t crc;
};

float currentTemp = 0.0;
float targetTemp = 22.0;
bool isBoilerOn = false;
bool lastTxSuccess = false;
String modeStr = "ROUTER";

char opMode[2] = "1"; // "1"=Router, "2"=AP
bool shouldSaveConfig = false;

void saveConfigCallback() {
    shouldSaveConfig = true;
}

void loadConfig() {
    if (LittleFS.begin()) {
        if (LittleFS.exists("/config.txt")) {
            File f = LittleFS.open("/config.txt", "r");
            if (f) {
                String val = f.readStringUntil('\n');
                val.trim();
                if (val == "1" || val == "2") {
                    strcpy(opMode, val.c_str());
                }
                f.close();
            }
        }
    }
}

void saveConfig() {
    File f = LittleFS.open("/config.txt", "w");
    if (f) {
        f.println(opMode);
        f.close();
    }
}

void setupWiFi() {
    WiFiManager wm;
    wm.setSaveConfigCallback(saveConfigCallback);

    String customHtml = String("type='hidden'><label>Mode</label><br><select name='opMode'>") +
                        "<option value='1'" + (String(opMode) == "1" ? " selected" : "") + ">ROUTER MODE</option>" +
                        "<option value='2'" + (String(opMode) == "2" ? " selected" : "") + ">AP MODE</option>" +
                        "</select><br";

    WiFiManagerParameter custom_opMode("opMode", "Mode", opMode, 2, customHtml.c_str());
    wm.addParameter(&custom_opMode);

    if (String(opMode) == "2") {
        modeStr = "AP MODE";
        WiFi.mode(WIFI_AP);
        WiFi.softAP("Boiler-Setup"); // Works as standalone AP
    } else {
        modeStr = "ROUTER";
        if (!wm.autoConnect("Boiler-Setup")) {
            ESP.restart();
        }
    }

    strcpy(opMode, custom_opMode.getValue());

    if (shouldSaveConfig) {
        saveConfig();
        if (String(opMode) == "2") {
            ESP.restart();
        }
    }
}

void drawSignalIcon(bool ok) {
    if (ok) {
        display.drawCircle(118, 12, 8, SSD1306_WHITE);
        display.drawCircle(118, 12, 5, SSD1306_WHITE);
        display.fillRect(110, 12, 16, 10, SSD1306_BLACK); // clear bottom part
        display.drawLine(118, 12, 118, 20, SSD1306_WHITE);
        display.fillRect(117, 20, 3, 3, SSD1306_WHITE);
    } else {
        display.drawLine(112, 6, 124, 18, SSD1306_WHITE);
        display.drawLine(124, 6, 112, 18, SSD1306_WHITE);
    }
}

void updateScreen() {
    display.clearDisplay();

    // Top Row
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("ESP-12F");

    int16_t x1, y1;
    uint16_t w, h;
    String centerText = "[" + modeStr + "]";
    display.getTextBounds(centerText, 0, 0, &x1, &y1, &w, &h);
    display.setCursor((128 - w) / 2, 0);
    display.print(centerText);

    drawSignalIcon(lastTxSuccess);

    // Center Row (Huge Text)
    display.setTextSize(3);
    String tempStr = String(currentTemp, 1);
    display.getTextBounds(tempStr, 0, 0, &x1, &y1, &w, &h);
    display.setCursor((128 - w) / 2, 20);
    display.print(tempStr);

    // Bottom Row
    display.setTextSize(1);
    display.setCursor(0, 56);
    display.print("Set: ");
    display.print(targetTemp, 1);
    display.print("C");

    String ipStr = WiFi.getMode() == WIFI_AP ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
    display.getTextBounds(ipStr, 0, 0, &x1, &y1, &w, &h);
    display.setCursor(128 - w, 56);
    display.print(ipStr);

    display.display();
}

void handleRoot() {
    String html = "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{background-color:#1a1a1a;color:white;font-family:sans-serif;text-align:center;padding:20px;}";
    html += "input[type=range]{width:80%;} .status{font-size:24px;margin-top:20px;}</style></head><body>";
    html += "<h2>Thermostat Control</h2>";
    html += "<h1>" + String(currentTemp, 1) + " &deg;C</h1>";
    html += "<p>Target: <span id='tval'>" + String(targetTemp, 1) + "</span> &deg;C</p>";
    html += "<input type='range' min='10' max='35' step='0.5' value='" + String(targetTemp, 1) + "' id='slider' oninput='document.getElementById(\"tval\").innerHTML=this.value' onchange='fetch(\"/set?t=\"+this.value)'>";
    html += "<div class='status'>Boiler: ";
    html += isBoilerOn ? "<span style='color:green;'>ON</span>" : "<span style='color:red;'>OFF</span>";
    html += "</div></body></html>";
    server.send(200, "text/html", html);
}

void handleSetTemp() {
    if (server.hasArg("t")) {
        targetTemp = server.arg("t").toFloat();
    }
    server.send(200, "text/plain", "OK");
}

void setup() {
    Serial.begin(115200);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    Wire.begin(OLED_SDA, OLED_SCL);
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println("SSD1306 init failed");
    }
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Booting...");
    display.display();

    sensors.begin();
    radio.begin();

    loadConfig();
    setupWiFi();

    server.on("/", HTTP_GET, handleRoot);
    server.on("/set", HTTP_GET, handleSetTemp);
    server.begin();
}

void loop() {
    static uint32_t lastRead = 0;
    static uint32_t btnPressTime = 0;
    static bool btnPressed = false;

    server.handleClient();

    if (digitalRead(BUTTON_PIN) == LOW) {
        if (!btnPressed) {
            btnPressed = true;
            btnPressTime = millis();
        } else if (millis() - btnPressTime >= 5000) {
            display.clearDisplay();
            display.setCursor(0, 0);
            display.print("Factory Reset...");
            display.display();

            WiFiManager wm;
            wm.resetSettings();
            LittleFS.format();
            delay(1000);
            ESP.restart();
        }
    } else {
        btnPressed = false;
    }

    if (millis() - lastRead >= 2000) {
        lastRead = millis();

        sensors.requestTemperatures();
        currentTemp = sensors.getTempCByIndex(0);

        if (currentTemp < targetTemp - 0.5) {
            isBoilerOn = true;
        } else if (currentTemp >= targetTemp) {
            isBoilerOn = false;
        }

        Packet pkt;
        pkt.cmd = isBoilerOn ? 1 : 0;
        pkt.temp = (int16_t)(currentTemp * 100);
        pkt.target = (int16_t)(targetTemp * 100);
        pkt.crc = pkt.cmd ^ (pkt.temp & 0xFF) ^ (pkt.target & 0xFF);

        lastTxSuccess = radio.send(&pkt, sizeof(pkt));

        updateScreen();
    }
}
