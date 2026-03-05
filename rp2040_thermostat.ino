#include <SPI.h>
#include <RF24.h>
#include <Adafruit_NeoPixel.h>

// --- Pins Setup ---
#define RELAY_PIN 0      // GP0
#define RGB_PIN 16       // GP16

#define NRF_SCK 2        // GP2
#define NRF_MOSI 3       // GP3
#define NRF_MISO 4       // GP4
#define NRF_CE 5         // GP5
#define NRF_CSN 6        // GP6

Adafruit_NeoPixel rgb(1, RGB_PIN, NEO_GRB + NEO_KHZ800);
RF24 radio(NRF_CE, NRF_CSN);

const byte address[6] = "THRM1";

struct Packet {
    uint8_t cmd;
    int16_t temp;
    int16_t target;
    uint8_t crc;
};

uint32_t lastPacketTime = 0;
bool isRelayOn = false;

void setLEDColor(uint8_t r, uint8_t g, uint8_t b) {
    rgb.setPixelColor(0, rgb.Color(r, g, b));
    rgb.show();
}

void setup() {
    Serial.begin(115200);

    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);

    rgb.begin();
    rgb.setBrightness(50);
    setLEDColor(0, 0, 255); // Blue while booting

    SPI.setSCK(NRF_SCK);
    SPI.setTX(NRF_MOSI);
    SPI.setRX(NRF_MISO);
    SPI.begin();

    if (!radio.begin(&SPI)) {
        Serial.println("Radio hardware not responding!");
        while (1) {
            setLEDColor(255, 0, 0); // Red error
            delay(500);
            setLEDColor(0, 0, 0);
            delay(500);
        }
    }

    radio.setChannel(76);
    radio.setDataRate(RF24_250KBPS);
    radio.setPALevel(RF24_PA_MAX);
    radio.openReadingPipe(1, address);
    radio.setPayloadSize(sizeof(Packet));
    radio.startListening();

    Serial.println("RP2040-Zero Receiver Started");
    setLEDColor(0, 255, 0); // Green indicating ready
}

void loop() {
    if (radio.available()) {
        Packet pkt;
        radio.read(&pkt, sizeof(pkt));

        uint8_t calcCrc = pkt.cmd ^ (pkt.temp & 0xFF) ^ (pkt.target & 0xFF);

        if (calcCrc == pkt.crc) {
            lastPacketTime = millis();

            if (pkt.cmd == 1) {
                isRelayOn = true;
                digitalWrite(RELAY_PIN, HIGH);
                setLEDColor(0, 255, 0); // Green
            } else if (pkt.cmd == 0) {
                isRelayOn = false;
                digitalWrite(RELAY_PIN, LOW);
                setLEDColor(0, 255, 0); // Green (Connection OK, simply OFF)
            }

            Serial.print("Received - Temp: ");
            Serial.print(pkt.temp / 100.0);
            Serial.print(" Target: ");
            Serial.print(pkt.target / 100.0);
            Serial.print(" Cmd: ");
            Serial.println(pkt.cmd);
        }
    }

    // Failsafe mechanism: 5 minutes = 300,000 ms
    if (millis() - lastPacketTime > 300000) {
        if (isRelayOn) {
            digitalWrite(RELAY_PIN, LOW);
            isRelayOn = false;
        }
        setLEDColor(255, 0, 0); // Red indicating connection lost
    }
}
