import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from deepface import DeepFace
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np

class FaceAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Analysis Tool")
        self.root.geometry("900x700")
        
        self.close_call_threshold = 0.06
        
        self.detectors = ['retinaface', 'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn']
        
        self.setup_ui()
        
    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Detection Model:").pack(side=tk.LEFT, padx=5)
        self.detector_var = tk.StringVar(value='retinaface')
        detector_combo = ttk.Combobox(control_frame, textvariable=self.detector_var, 
                                      values=self.detectors, state='readonly', width=15)
        detector_combo.pack(side=tk.LEFT, padx=5)
        
        self.select_btn = ttk.Button(control_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=20)
        
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Faces", 
                                      command=self.analyze_image, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(control_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        image_frame = ttk.LabelFrame(self.root, text="Selected Image", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.image_label = ttk.Label(image_frame, text="No image selected", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        results_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.results_text.tag_config("warning", foreground="orange", font=("Arial", 10, "bold"))
        self.results_text.tag_config("header", foreground="blue", font=("Arial", 11, "bold"))
        self.results_text.tag_config("error", foreground="red", font=("Arial", 10, "bold"))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.current_image_path = None
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.original_image = Image.open(file_path)
            self.display_image(self.original_image)
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
    def display_image(self, image):
        try:
            display_img = image.copy()
            display_img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {str(e)}")
            
    def check_close_call(self, predictions, category_name):
        if not predictions or len(predictions) < 2:
            return False, None
            
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        first = sorted_preds[0]
        second = sorted_preds[1]
        
        difference = first[1] - second[1]
        
        if difference <= self.close_call_threshold * 100:
            return True, {
                'first': first,
                'second': second,
                'difference': difference
            }
        
        return False, None
        
    def analyze_image(self):
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please select an image first")
            return
            
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Analyzing image...")
        self.root.update()
        
        try:
            detector = self.detector_var.get()
            results = DeepFace.analyze(
                img_path=self.current_image_path, 
                actions=["race"], 
                detector_backend=detector,
                enforce_detection=False
            )
            
            annotated_image = self.draw_face_boxes(self.original_image, results)
            self.display_image(annotated_image)
            
            self.display_results(results)
            self.status_var.set(f"Analysis complete - Found {len(results)} face(s)")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error during analysis:\n{str(e)}\n\n", "error")
            self.results_text.insert(tk.END, "Try a different detection model or check if faces are clearly visible.", "error")
            self.status_var.set("Analysis failed")
    
    def draw_face_boxes(self, image, results):
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        try:
            font = ImageFont.truetype("arial.ttf", 40)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        face_num = 0
        for face_data in results:
            face_num += 1
            
            region = face_data.get('region', {})
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            
            if w > 0 and h > 0:
                for thickness in range(5):
                    draw.rectangle(
                        [(x - thickness, y - thickness), 
                         (x + w + thickness, y + h + thickness)],
                        outline='lime',
                        width=2
                    )
                
                label = f"Face #{face_num}"
                
                bbox = draw.textbbox((0, 0), label, font=small_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                label_x = x
                label_y = max(0, y - text_height - 10)
                
                draw.rectangle(
                    [(label_x - 5, label_y - 5),
                     (label_x + text_width + 10, label_y + text_height + 5)],
                    fill='lime',
                    outline='black',
                    width=2
                )
                
                draw.text((label_x, label_y), label, fill='black', font=small_font)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                num_str = str(face_num)
                num_bbox = draw.textbbox((0, 0), num_str, font=font)
                num_width = num_bbox[2] - num_bbox[0]
                num_height = num_bbox[3] - num_bbox[1]
                
                circle_radius = max(num_width, num_height) // 2 + 10
                draw.ellipse(
                    [(center_x - circle_radius, center_y - circle_radius),
                     (center_x + circle_radius, center_y + circle_radius)],
                    fill='yellow',
                    outline='black',
                    width=3
                )
                
                draw.text(
                    (center_x - num_width // 2, center_y - num_height // 2),
                    num_str,
                    fill='black',
                    font=font
                )
        
        return img_with_boxes
            
    def display_results(self, results):
        if not results:
            self.results_text.insert(tk.END, "No faces detected in the image.\n")
            return
            
        face_count = 0
        faces_needing_verification = []
        
        for face_data in results:
            face_count += 1
            
            self.results_text.insert(tk.END, f"\n{'='*60}\n", "header")
            self.results_text.insert(tk.END, f"FACE #{face_count}\n", "header")
            self.results_text.insert(tk.END, f"{'='*60}\n\n", "header")
            
            needs_verification = False
            
            self.results_text.insert(tk.END, "RACE ANALYSIS:\n", "header")
            race_data = face_data.get('race', {})
            dominant_race = face_data.get('dominant_race', 'Unknown')
            
            self.results_text.insert(tk.END, f"  Dominant: {dominant_race}\n")
            self.results_text.insert(tk.END, "  Confidence scores:\n")
            
            for race, score in sorted(race_data.items(), key=lambda x: x[1], reverse=True):
                self.results_text.insert(tk.END, f"    - {race}: {score:.2f}%\n")
            
            is_close, close_data = self.check_close_call(race_data, "race")
            if is_close:
                needs_verification = True
                self.results_text.insert(tk.END, f"\n  ⚠ WARNING: Close call detected!\n", "warning")
                self.results_text.insert(tk.END, f"    {close_data['first'][0]}: {close_data['first'][1]:.2f}% vs ", "warning")
                self.results_text.insert(tk.END, f"{close_data['second'][0]}: {close_data['second'][1]:.2f}%\n", "warning")
                self.results_text.insert(tk.END, f"    Difference: only {close_data['difference']:.2f}%\n", "warning")
                self.results_text.insert(tk.END, "    → HUMAN VERIFICATION RECOMMENDED\n\n", "warning")
            
            if needs_verification:
                faces_needing_verification.append(face_count)
        
        self.results_text.insert(tk.END, f"\n{'='*60}\n", "header")
        self.results_text.insert(tk.END, f"SUMMARY\n", "header")
        self.results_text.insert(tk.END, f"{'='*60}\n", "header")
        self.results_text.insert(tk.END, f"Total faces detected: {face_count}\n")
        
        if faces_needing_verification:
            self.results_text.insert(tk.END, f"\nFaces requiring human verification: {len(faces_needing_verification)}\n", "warning")
            self.results_text.insert(tk.END, f"Face numbers: {', '.join(map(str, faces_needing_verification))}\n", "warning")
        else:
            self.results_text.insert(tk.END, "\nAll predictions have high confidence.\n")
            
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        if hasattr(self, 'original_image'):
            self.display_image(self.original_image)
        self.status_var.set("Results cleared")

def main():
    root = tk.Tk()
    app = FaceAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()


