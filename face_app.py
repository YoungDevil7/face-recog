import cv2
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Initialize variables
        self.camera = None
        self.is_capturing = False
        self.current_user = ""
        self.captured_images = 0
        self.total_images_to_capture = 50  # Increased to 50 images
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=100
        )

        # Load or create users database
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=10, font=('Helvetica', 12))
        self.style.configure("TLabel", font=('Helvetica', 12))
        self.style.configure("TEntry", font=('Helvetica', 12))

        # Create video frame
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Camera Feed", padding="10")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Create control frame
        self.control_frame = ttk.Frame(self.main_frame, padding="10")
        self.control_frame.pack(fill=tk.X, pady=10)

        # Create input field
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(self.control_frame, textvariable=self.name_var, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        # Create buttons
        self.register_btn = ttk.Button(self.control_frame, text="Start Registration", command=self.start_registration)
        self.register_btn.pack(side=tk.LEFT, padx=5)

        self.capture_btn = ttk.Button(self.control_frame, text="Capture Now", command=self.manual_capture, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(self.control_frame, text="Stop Registration", command=self.stop_registration, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.train_btn = ttk.Button(self.control_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.recognize_btn = ttk.Button(self.control_frame, text="Start Recognition", command=self.start_recognition)
        self.recognize_btn.pack(side=tk.LEFT, padx=5)

        # Create status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, font=('Helvetica', 12))
        self.status_label.pack(pady=10)

        # Create user list
        self.user_frame = ttk.LabelFrame(self.main_frame, text="Registered Users", padding="10")
        self.user_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.update_user_list()

    def update_user_list(self):
        # Clear existing widgets
        for widget in self.user_frame.winfo_children():
            widget.destroy()

        # Create new list
        for user_id, name in self.users.items():
            user_label = ttk.Label(self.user_frame, text=f"ID: {user_id} - {name}")
            user_label.pack(anchor=tk.W, pady=2)

    def start_registration(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        self.current_user = name
        self.captured_images = 0
        self.is_capturing = True

        # Create user directory
        user_dir = os.path.join('dataset', name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # Start camera with DirectShow backend
        self.camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please make sure it's not in use by another application.")
            self.is_capturing = False
            return

        # Enable/disable buttons
        self.register_btn.configure(state=tk.DISABLED)
        self.capture_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL)
        self.train_btn.configure(state=tk.DISABLED)
        self.recognize_btn.configure(state=tk.DISABLED)

        threading.Thread(target=self.capture_frames, daemon=True).start()

    def stop_registration(self):
        self.is_capturing = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None

        # Reset button states
        self.register_btn.configure(state=tk.NORMAL)
        self.capture_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.DISABLED)
        self.train_btn.configure(state=tk.NORMAL)
        self.recognize_btn.configure(state=tk.NORMAL)

        self.status_var.set(f"Registration stopped. Captured {self.captured_images} images.")
        self.update_user_list()

    def manual_capture(self):
        if not self.is_capturing or self.camera is None:
            return

        ret, frame = self.camera.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Add padding around the face
            padding = int(w * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract and process face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)
            
            # Save the processed face
            cv2.imwrite(f'dataset/{self.current_user}/{self.captured_images}.jpg', face_roi)
            self.captured_images += 1
            self.status_var.set(f"Captured {self.captured_images} images")
        else:
            messagebox.showwarning("Warning", "No face detected!")

    def capture_frames(self):
        last_capture_time = 0
        capture_interval = 1.0  # Capture every second
        
        while self.is_capturing:
            ret, frame = self.camera.read()
            if not ret:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Convert frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.configure(image=photo)
            self.video_label.image = photo

            # Auto-capture with time interval
            current_time = time.time()
            if (self.captured_images < self.total_images_to_capture and 
                current_time - last_capture_time >= capture_interval and 
                len(faces) > 0):
                
                # Get the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Add padding around the face
                padding = int(w * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                # Extract and process face
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_roi = cv2.equalizeHist(face_roi)
                
                # Save the processed face
                cv2.imwrite(f'dataset/{self.current_user}/{self.captured_images}.jpg', face_roi)
                self.captured_images += 1
                last_capture_time = current_time
                self.status_var.set(f"Captured {self.captured_images} images")

            if self.captured_images >= self.total_images_to_capture:
                self.stop_registration()
                break

    def train_model(self):
        try:
            faces = []
            labels = []
            label_id = 0
            label_map = {}
            
            # First, create a mapping of names to IDs
            for name in os.listdir('dataset'):
                if os.path.isdir(os.path.join('dataset', name)):
                    label_map[name] = label_id
                    label_id += 1
            
            # Then collect all face images
            for name, label in label_map.items():
                user_dir = os.path.join('dataset', name)
                if not os.path.exists(user_dir):
                    continue

                for image_name in os.listdir(user_dir):
                    if image_name.endswith('.jpg'):
                        image_path = os.path.join(user_dir, image_name)
                        face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            # Apply histogram equalization
                            face_img = cv2.equalizeHist(face_img)
                            faces.append(face_img)
                            labels.append(label)

            if not faces:
                messagebox.showerror("Error", "No training data found")
                return

            print(f"Training with {len(faces)} images from {len(label_map)} users")
            
            # Train the model
            self.face_recognizer.train(faces, np.array(labels))
            self.face_recognizer.save('classifier.xml')
            
            # Update users.json with the new mapping
            self.users = {str(label): name for name, label in label_map.items()}
            with open('users.json', 'w') as f:
                json.dump(self.users, f)
            
            messagebox.showinfo("Success", "Model trained successfully!")
            self.update_user_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def start_recognition(self):
        if not os.path.exists('classifier.xml'):
            messagebox.showerror("Error", "Please train the model first")
            return

        self.is_capturing = True
        # Start camera with DirectShow backend
        self.camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please make sure it's not in use by another application.")
            self.is_capturing = False
            return

        threading.Thread(target=self.recognize_frames, daemon=True).start()

    def recognize_frames(self):
        while self.is_capturing:
            ret, frame = self.camera.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Add padding around the face
                padding = int(w * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding)
                
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_roi = cv2.equalizeHist(face_roi)
                
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    if confidence < 60:  # Increased threshold for better accuracy
                        name = self.users.get(str(label), "Unknown")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({confidence:.2f}%)", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Error", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.configure(image=photo)
            self.video_label.image = photo

    def __del__(self):
        if self.camera is not None:
            self.camera.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop() 