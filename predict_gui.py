import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
import os
from tkinter import messagebox
import json  

class SignRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Recognizer")
        self.root.geometry("1024x600")
        self.root.minsize(800, 600)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.model = load_model('traffic_sign_model.h5')
        
        self.meta_df = pd.read_csv('Meta.csv')
        
        with open('traffic_signs.json', 'r', encoding='utf-8') as f:
            self.signs_info = {item['id']: item['znak'] for item in json.load(f)}
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.grid(row=0, column=0, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.main_frame.columnconfigure(0, weight=4)  
        self.main_frame.columnconfigure(1, weight=6)  
        self.main_frame.rowconfigure(0, weight=1)
        
        self.left_frame.columnconfigure(0, weight=1)
        self.left_frame.columnconfigure(1, weight=1)
        self.left_frame.columnconfigure(2, weight=1)
        
        self.current_image = None
        self.photo_image = None
        self.gallery_images = [] 
        
        self.create_widgets()
        
    def create_widgets(self):
        self.sign_name_label = ttk.Label(self.left_frame, text="", font=('Arial', 12, 'bold'))
        self.sign_name_label.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        self.image_frame = ttk.Frame(self.left_frame)
        self.image_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        self.image_frame.columnconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="Drag an image here or click to select")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.select_btn = ttk.Button(self.left_frame, text="Select Image", command=self.select_image)
        self.select_btn.grid(row=2, column=0, pady=5)
        
        self.browse_folder_btn = ttk.Button(self.left_frame, text="Browse Folder", command=self.browse_folder)
        self.browse_folder_btn.grid(row=2, column=1, pady=5)
        
        self.predict_btn = ttk.Button(self.left_frame, text="Recognize Signs", command=self.predict_signs)
        self.predict_btn.grid(row=2, column=2, pady=5)
        
        self.results_frame = ttk.Frame(self.left_frame)
        self.results_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.result_text = tk.Text(self.results_frame, height=10, width=40)
        self.result_text.grid(row=0, column=0, padx=5)
        
        self.ref_images_frame = ttk.Frame(self.results_frame)
        self.ref_images_frame.grid(row=0, column=1, padx=5)
        
        self.gallery_canvas = tk.Canvas(self.right_frame)
        self.gallery_scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", 
                                             command=self.gallery_canvas.yview)
        self.gallery_frame = ttk.Frame(self.gallery_canvas)
        
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)
        
        self.gallery_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.gallery_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.gallery_window = self.gallery_canvas.create_window((0, 0), 
                                                              window=self.gallery_frame,
                                                              anchor="nw")
        self.gallery_frame.bind("<Configure>", self.on_frame_configure)
        self.gallery_canvas.bind("<Configure>", self.on_canvas_configure)
        
        self.gallery_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label.drop_target_register('DND_Files')
        self.image_label.dnd_bind('<<Drop>>', self.handle_drop)
    
    def on_frame_configure(self, event=None):
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        
    def on_canvas_configure(self, event):
        self.gallery_canvas.itemconfig(self.gallery_window, width=event.width)
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.load_and_display_image(file_path)
    
    def handle_drop(self, event):
        file_path = event.data
        self.load_and_display_image(file_path)
    def load_and_display_image(self, file_path):
        image = Image.open(file_path)
        image = image.convert('RGB') 
        width, height = image.size
        if width > height:
            new_width = 128
            new_height = int(128 * height / width)
        else:
            new_height = 128
            new_width = int(128 * width / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        background = Image.new('RGB', (128, 128), 'white')
        offset = ((128 - new_width) // 2, (128 - new_height) // 2)
        background.paste(image, offset)
        self.photo_image = ImageTk.PhotoImage(background)
        self.image_label.configure(image=self.photo_image, text="")
        
        self.sign_name_label.configure(text="")
        
        self.current_image = file_path
    
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        sign_images = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            sign = img[y:y+h, x:x+w]
            sign = cv2.resize(sign, (64, 64))
            sign = sign / 255.0
            sign_images.append(sign)

        if not sign_images:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            sign_images.append(img)
            
        return np.array(sign_images)    
    
    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if folder_path:
            for widget in self.gallery_frame.winfo_children():
                widget.destroy()
            
            image_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            self.gallery_images.clear()
            
            self.gallery_frame.update_idletasks()  
            num_columns = max(3, self.gallery_canvas.winfo_width() // 150) 
            for i, img_file in enumerate(image_files):
                try:
                    img_path = os.path.join(folder_path, img_file)
                    img = Image.open(img_path)
                    img.thumbnail((150, 150))
                    photo = ImageTk.PhotoImage(img)
                    self.gallery_images.append(photo)  
                    
                    btn = ttk.Button(self.gallery_frame, image=photo, 
                                   command=lambda p=img_path: self.select_gallery_image(p))
                    row = i // num_columns
                    col = i % num_columns
                    btn.grid(row=row, column=col, padx=5, pady=5)
                    
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")
    
    def select_gallery_image(self, image_path):
        self.load_and_display_image(image_path)
    
    def predict_signs(self):
        if self.current_image is None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Please select an image first.")
            return
            
        self.result_text.delete(1.0, tk.END)
        for widget in self.ref_images_frame.winfo_children():
            widget.destroy()
        
        signs = self.preprocess_image(self.current_image)
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Detected Signs:\n\n")
        
        highest_confidence_name = None
        highest_confidence = 0
        
        for i, sign in enumerate(signs):
            sign = np.expand_dims(sign, axis=0)
            
            predictions = self.model.predict(sign, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            sign_info = self.meta_df[self.meta_df['ClassId'] == predicted_class]
            sign_id = sign_info['SignId'].values[0] if not sign_info.empty else "Unknown"
            
            sign_name = self.signs_info.get(predicted_class, "Unknown Sign")
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                highest_confidence_name = sign_name
            
            self.result_text.insert(tk.END, f"Sign {i+1}:\n")
            self.result_text.insert(tk.END, f"Sign Name: {sign_name}\n")
            self.result_text.insert(tk.END, f"Class ID: {predicted_class}\n")
            self.result_text.insert(tk.END, f"Sign ID: {sign_id}\n")
            self.result_text.insert(tk.END, f"Confidence: {confidence:.2%}\n\n")
            
            if not sign_info.empty:
                meta_img_path = sign_info['Path'].values[0]
                try:
                    ref_img = Image.open(meta_img_path)
                    ref_img.thumbnail((100, 100))  
                    photo = ImageTk.PhotoImage(ref_img)
                    
                    img_label = ttk.Label(self.ref_images_frame, image=photo, text=f"Reference {i+1}")
                    img_label.image = photo
                    img_label.grid(row=i, column=0, pady=5)
                except Exception as e:
                    print(f"Error loading reference image: {e}")
        
        if highest_confidence_name:
            self.sign_name_label.configure(text=highest_confidence_name)

    def _on_mousewheel(self, event):
        if event.delta:
            self.gallery_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            if event.num == 4:
                self.gallery_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.gallery_canvas.yview_scroll(1, "units")

def main():
    root = TkinterDnD.Tk()
    app = SignRecognizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
