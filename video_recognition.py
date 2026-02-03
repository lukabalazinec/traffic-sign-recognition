import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import pandas as pd
import json
from datetime import datetime

class VideoRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Video Recognition")
        
        self.model = load_model('traffic_sign_model.h5')
        self.meta_df = pd.read_csv('Meta.csv')
        
        with open('traffic_signs.json', 'r', encoding='utf-8') as f:
            self.signs_info = {item['id']: item['znak'] for item in json.load(f)}
        
        self.cap = None
        self.is_playing = False
        self.is_webcam = False
        self.is_paused = False
        self.last_frame = None
        
        self.high_confidence_detections = {} 
        
        self.create_widgets()
        
    def create_widgets(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.grid(row=0, column=0, rowspan=3, padx=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.grid(row=0, column=0)
        self.video_label.bind('<Button-1>', self.toggle_pause)
        
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.select_btn = ttk.Button(self.control_panel, text="Select Video", command=self.select_video)
        self.select_btn.grid(row=0, column=0, pady=5, padx=5)
        
        self.webcam_btn = ttk.Button(self.control_panel, text="Use Webcam", command=self.start_webcam)
        self.webcam_btn.grid(row=0, column=1, pady=5, padx=5)
        
        self.start_btn = ttk.Button(self.control_panel, text="Start", command=self.start_video, state='disabled')
        self.start_btn.grid(row=1, column=0, pady=5, padx=5)
        
        self.stop_btn = ttk.Button(self.control_panel, text="Stop", command=self.stop_video, state='disabled')
        self.stop_btn.grid(row=1, column=1, pady=5, padx=5)
        
        self.results_frame = ttk.Frame(self.control_panel)
        self.results_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.current_frame = ttk.LabelFrame(self.results_frame, text="Current Detections")
        self.current_frame.grid(row=0, column=0, padx=5, pady=5)
        self.result_text = tk.Text(self.current_frame, height=10, width=40)
        self.result_text.grid(row=0, column=0, pady=5, padx=5)
        
        self.high_conf_frame = ttk.LabelFrame(self.results_frame, text="High Confidence Detections (>96%)")
        self.high_conf_frame.grid(row=1, column=0, padx=5, pady=5)
        self.high_conf_text = tk.Text(self.high_conf_frame, height=10, width=40)
        self.high_conf_text.grid(row=0, column=0, pady=5, padx=5)    
    def toggle_pause(self, event=None):
        if self.is_playing:
            self.is_paused = not self.is_paused
            if not self.is_paused:
                self.update_frame()

    def start_webcam(self):
        if self.cap is not None:
            self.stop_video()
        
        self.cap = cv2.VideoCapture(0)  
        if self.cap.isOpened():
            self.is_webcam = True
            self.start_btn.configure(state='normal')
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Webcam initialized. Click Start to begin processing.\n")
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Error: Could not access webcam.\n")

    def select_video(self):
        if self.cap is not None:
            self.stop_video()
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_webcam = False
                self.start_btn.configure(state='normal')
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Video loaded successfully. Click Start to begin processing.\n")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_signs = []
        high_confidence_signs = []

        for contour in contours:
            if cv2.contourArea(contour) < 1000: 
                continue
            x, y, w, h = cv2.boundingRect(contour)
            sign = rgb_frame[y:y+h, x:x+w]
            sign = cv2.resize(sign, (64, 64))
            sign = sign / 255.0
            prediction = self.model.predict(np.expand_dims(sign, axis=0), verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            if confidence > 0.7:  
                color = (0, 255, 0) if confidence > 0.96 else (0, 255, 255)  
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                sign_info = self.meta_df[self.meta_df['ClassId'] == predicted_class]
                sign_name = self.signs_info.get(predicted_class, "Unknown Sign")

                if not sign_info.empty:
                    sign_id = sign_info['SignId'].values[0]
                    text = f"{sign_name} ({confidence:.1%})"
                    words = text.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) > 20: 
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            current_line.append(word)
                    if current_line:
                        lines.append(' '.join(current_line))

                    for i, line in enumerate(lines):
                        y_pos = y - 10 - (15 * (len(lines) - i - 1))  
                        cv2.putText(frame, line, (x, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    detected_signs.append((sign_name, confidence))
                    
                    if confidence > 0.96:
                        high_confidence_signs.append((sign_name, confidence))
                        if sign_name not in self.high_confidence_detections or confidence > self.high_confidence_detections[sign_name][0]:
                            self.high_confidence_detections[sign_name] = (confidence, pd.Timestamp.now())

        return frame, detected_signs, high_confidence_signs

    def update_frame(self):
        if self.is_playing and self.cap is not None:
            if not self.is_paused:
                ret, frame = self.cap.read()
                if ret:
                    self.last_frame, detected_signs, high_confidence_signs = self.process_frame(frame)
                    self.update_display(self.last_frame, detected_signs, high_confidence_signs)
                else:
                    self.stop_video()
                    return
            else:
                if self.last_frame is not None:
                    img = Image.fromarray(cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB))
                    aspect_ratio = img.width / img.height
                    new_height = 600
                    new_width = int(aspect_ratio * new_height)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    self.video_label.image = ImageTk.PhotoImage(image=img)
                    self.video_label.configure(image=self.video_label.image)
            if not self.is_paused:
                self.root.after(30, self.update_frame)  

    def update_display(self, processed_frame, detected_signs, high_confidence_signs):
        self.result_text.delete(1.0, tk.END)
        if detected_signs:
            self.result_text.insert(tk.END, "Current Detections:\n\n")
            for i, (sign_name, conf) in enumerate(detected_signs, 1):
                self.result_text.insert(tk.END, f"{i}. {sign_name}\n")
                self.result_text.insert(tk.END, f"   Confidence: {conf:.1%}\n\n")
        
        self.high_conf_text.delete(1.0, tk.END)
        self.high_conf_text.insert(tk.END, "Best Detections (>96%):\n\n")
        
        sorted_detections = sorted(
            self.high_confidence_detections.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        
        for i, (sign_name, (conf, timestamp)) in enumerate(sorted_detections, 1):
            self.high_conf_text.insert(tk.END, f"{i}. {sign_name}\n")
            self.high_conf_text.insert(tk.END, f"   Confidence: {conf:.1%}\n")
            self.high_conf_text.insert(tk.END, f"   Detected at: {timestamp.strftime('%H:%M:%S')}\n\n")
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_frame)
        aspect_ratio = img.width / img.height
        new_height = 600
        new_width = int(aspect_ratio * new_height)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def start_video(self):
        if self.cap is not None and not self.is_playing:
            self.is_playing = True
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
            self.update_frame()
    
    def stop_video(self):
        self.is_playing = False
        self.is_paused = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='disabled')
        self.video_label.configure(image='')
        self.last_frame = None
        self.high_confidence_detections.clear()

def main():
    root = tk.Tk()
    app = VideoRecognizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
