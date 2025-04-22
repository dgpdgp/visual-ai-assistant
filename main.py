import cv2
from ultralytics import YOLO
import pyttsx3
import tkinter as tk

class VisualAssistant:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.engine = pyttsx3.init()
        self.root = tk.Tk()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Визуальный помощник")
        tk.Button(self.root, text="Старт", command=self.run).pack()
        tk.Button(self.root, text="Стоп", command=self.stop).pack()
        
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            results = self.model(frame)
            for obj in results[0].boxes:
                label = self.model.names[int(obj.cls)]
                self.engine.say(f"Я вижу {label}")
                self.engine.runAndWait()
            cv2.imshow("Камера", frame)
            if cv2.waitKey(1) == 27:  # ESC для выхода
                break
        cap.release()

    def stop(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VisualAssistant()
    app.root.mainloop()