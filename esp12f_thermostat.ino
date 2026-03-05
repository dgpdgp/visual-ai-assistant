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

    void init() {
        pinMode(pin_sck, OUTPUT);
        pinMode(pin_mosi, OUTPUT);
        pinMode(pin_miso, INPUT);
        pinMode(pin_ce, OUTPUT);
        pinMode(pin_csn, OUTPUT);

        digitalWrite(pin_sck, LOW);
        digitalWrite(pin_ce, LOW);
        digitalWrite(pin_csn, HIGH);

        delay(100);
        writeRegister(0x00, 0x0E); // CONFIG: PWR_UP=1, PRIM_RX=0, EN_CRC=1, CRCO=1 (2-byte CRC)
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

// State Variables
float currentTemp = 0.0;
float targetTemp = 22.0;
bool isBoilerOn = false;
bool lastTxSuccess = false;
String modeStr = "ROUTER";

char opMode[2] = "1"; // "1"=Router, "2"=AP
bool shouldSaveConfig = false;

// Statistics & Logic
uint32_t lastStateChangeTime = 0;
uint32_t totalBoilerRunTime = 0; // ms
uint32_t lastBoilerStartTime = 0;
uint32_t boilerOnCount = 0;
uint32_t lastPacketTime = 0;

// History for chart (last 20 readings)
#define HISTORY_LEN 20
float tempHistory[HISTORY_LEN] = {0};
uint8_t historyIdx = 0;

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
        WiFi.softAP("Boiler-Setup");
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
        display.fillRect(110, 12, 16, 10, SSD1306_BLACK);
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
    String tempStr = String(currentTemp, 1) + "C";
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
    uint32_t uptime = millis() / 1000;
    uint32_t h = uptime / 3600;
    uint32_t m = (uptime % 3600) / 60;
    uint32_t s = uptime % 60;

    uint32_t runTimeSec = totalBoilerRunTime / 1000;
    if (isBoilerOn) {
        runTimeSec += (millis() - lastBoilerStartTime) / 1000;
    }
    uint32_t rth = runTimeSec / 3600;
    uint32_t rtm = (runTimeSec % 3600) / 60;

    String ipStr = WiFi.getMode() == WIFI_AP ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
    String stateStr = lastTxSuccess ? (isBoilerOn ? "<span style='color:orange;'>HEATING</span>" : "<span style='color:gray;'>IDLE</span>") : "<span style='color:red;'>NO SIGNAL</span>";
    String signalStr = lastTxSuccess ? "<span style='color:green;'>Good</span>" : "<span style='color:red;'>Lost</span>";
    uint32_t lastPktAgo = (millis() - lastPacketTime) / 1000;

    String html = "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{background-color:#1a1a1a;color:white;font-family:sans-serif;text-align:center;padding:10px;margin:0;}";
    html += "input[type=range]{width:90%;height:30px;} .card{background:#2a2a2a;padding:15px;border-radius:10px;margin-bottom:15px;}";
    html += "h1{margin:10px 0;font-size:48px;} .info{font-size:14px;color:#bbb;}</style></head><body>";

    html += "<h2>Thermostat v5.1</h2>";

    html += "<div class='card'>";
    html += "<h1>" + String(currentTemp, 1) + " &deg;C</h1>";
    html += "<div style='font-size:24px;margin:10px 0;'>" + stateStr + "</div>";
    html += "<p>Target: <span id='tval' style='font-size:20px;font-weight:bold;'>" + String(targetTemp, 1) + "</span> &deg;C</p>";
    html += "<input type='range' min='10' max='30' step='0.5' value='" + String(targetTemp, 1) + "' id='slider' oninput='document.getElementById(\"tval\").innerHTML=this.value' onchange='fetch(\"/set?t=\"+this.value)'>";
    html += "</div>";

    html += "<div class='card' style='text-align:left;'>";
    html += "<h3>Statistics</h3>";
    html += "<div class='info'>";
    html += "Uptime: " + String(h) + "h " + String(m) + "m " + String(s) + "s<br>";
    html += "IP Address: " + ipStr + "<br>";
    html += "Radio Link: " + signalStr + " (Last Pkt: " + String(lastPktAgo) + "s ago)<br>";
    html += "Boiler Runs: " + String(boilerOnCount) + "<br>";
    html += "Total Run Time: " + String(rth) + "h " + String(rtm) + "m<br>";
    html += "</div></div>";

    // Simple ASCII/SVG sparkline for temperature history
    html += "<div class='card'><h3>Temp History (Last 20)</h3>";
    html += "<svg width='100%' height='60' style='background:#111;border:1px solid #444;'>";
    html += "<polyline fill='none' stroke='#00ff00' stroke-width='2' points='";

    float minT = 100.0, maxT = -100.0;
    for(int i=0; i<HISTORY_LEN; i++) {
        if(tempHistory[i] != 0) {
            if(tempHistory[i] < minT) minT = tempHistory[i];
            if(tempHistory[i] > maxT) maxT = tempHistory[i];
        }
    }
    if (maxT == minT) { maxT += 1; minT -= 1; }

    for(int i=0; i<HISTORY_LEN; i++) {
        int realIdx = (historyIdx + i) % HISTORY_LEN;
        if(tempHistory[realIdx] != 0) {
            int x = (i * 100) / (HISTORY_LEN-1);
            int y = 60 - ((tempHistory[realIdx] - minT) * 60 / (maxT - minT));
            html += String(x) + "%," + String(y) + " ";
        }
    }
    html += "'/></svg></div>";

    html += "<script>setInterval(()=>location.reload(), 10000);</script>";
    html += "</body></html>";
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
    display.print("Booting v5.1...");
    display.display();

    sensors.begin();
    radio.init();

    loadConfig();
    setupWiFi();

    server.on("/", HTTP_GET, handleRoot);
    server.on("/set", HTTP_GET, handleSetTemp);
    server.begin();

    lastStateChangeTime = millis() - 180000; // Allow immediate first turn on/off
}

void loop() {
    static uint32_t lastRead = 0;
    static uint32_t lastHist = 0;
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

        // Save to history every 30s
        if (millis() - lastHist >= 30000) {
            lastHist = millis();
            tempHistory[historyIdx] = currentTemp;
            historyIdx = (historyIdx + 1) % HISTORY_LEN;
        }

        // Hysteresis & Minimum Run/Pause Time Logic (3 mins = 180000 ms)
        uint32_t timeSinceLastChange = millis() - lastStateChangeTime;
        bool wantBoilerOn = isBoilerOn;

        if (currentTemp <= targetTemp - 0.5) {
            wantBoilerOn = true;
        } else if (currentTemp >= targetTemp + 0.5) {
            wantBoilerOn = false;
        }

        if (wantBoilerOn != isBoilerOn && timeSinceLastChange >= 180000) {
            isBoilerOn = wantBoilerOn;
            lastStateChangeTime = millis();

            if (isBoilerOn) {
                boilerOnCount++;
                lastBoilerStartTime = millis();
            } else {
                totalBoilerRunTime += (millis() - lastBoilerStartTime);
            }
        }

        Packet pkt;
        pkt.cmd = isBoilerOn ? 1 : 0;
        pkt.temp = (int16_t)(currentTemp * 10);
        pkt.target = (int16_t)(targetTemp * 10);
        pkt.crc = pkt.cmd ^ (pkt.temp & 0xFF) ^ (pkt.target & 0xFF);

        lastTxSuccess = radio.send(&pkt, sizeof(pkt));
        if (lastTxSuccess) {
            lastPacketTime = millis();
        }

        updateScreen();
    }
}
