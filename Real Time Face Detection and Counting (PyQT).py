import sys
import cv2
import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QCheckBox, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from PyQt5.QtGui import QColor
import time
from PyQt5.QtWidgets import QRadioButton, QButtonGroup
from PyQt5.QtWidgets import QFileDialogt

class FaceCountPlot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(FaceCountPlot, self).__init__(fig)
        self.setParent(parent)
        self.x_data = []
        self.y_data = []
        self.start_time = None
        self.time_window = 10  # Display last 10 seconds

    def update_plot(self, count):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time
        self.x_data.append(elapsed_time)
        self.y_data.append(count)

        # Remove data points outside the time window
        while self.x_data and self.x_data[0] < elapsed_time - self.time_window:
            self.x_data.pop(0)
            self.y_data.pop(0)

        self.axes.clear()
        self.axes.plot(self.x_data, self.y_data)
        
        # Set x-axis limits to create scrolling effect
        x_min = max(0, elapsed_time - self.time_window)
        x_max = elapsed_time
        self.axes.set_xlim(x_min, x_max)
        
        # Set y-axis limits
        y_max = max(self.y_data) if self.y_data else 1
        self.axes.set_ylim(0, y_max + 1)

        self.axes.set_xlabel('Time (seconds)')
        self.axes.set_ylabel('Face Count')
        
        # Set y-axis to show only whole numbers
        self.axes.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Format x-axis to show seconds
        self.axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.1f}s"))
        
        self.draw()

    def reset(self):
        self.x_data = []
        self.y_data = []
        self.start_time = None
        self.axes.clear()
        self.draw()

def apply_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    return noisy

class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Real-time Face Detection")
        self.setGeometry(100, 100, 1600, 1000)
        self.current_frame = None  # Add this line to store the current frame

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Image Degradation controls
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)

# Input selection group
        input_selection_group = QGroupBox("Input Selection")
        input_selection_layout = QVBoxLayout()
        input_selection_group.setLayout(input_selection_layout)
        left_layout.addWidget(input_selection_group)

        # Radio buttons for input selection
        self.webcam_radio = QRadioButton("Webcam")
        self.clipboard_radio = QRadioButton("Clipboard Image")
        self.local_file_radio = QRadioButton("Local File")  # New radio button
        self.webcam_radio.setChecked(True)  # Default to webcam

        # Button group for radio buttons
        input_button_group = QButtonGroup(self)
        input_button_group.addButton(self.webcam_radio)
        input_button_group.addButton(self.clipboard_radio)
        input_button_group.addButton(self.local_file_radio)  # Add to button group

        # Start button
        self.start_input_button = QPushButton("Start Input")
        self.start_input_button.clicked.connect(self.start_input)

        # Add widgets to input selection layout
        input_selection_layout.addWidget(self.webcam_radio)
        input_selection_layout.addWidget(self.clipboard_radio)
        input_selection_layout.addWidget(self.local_file_radio)  # Add to layout
        input_selection_layout.addWidget(self.start_input_button)

        image_degradation_group = QGroupBox("Image Degradation")
        image_degradation_layout = QVBoxLayout()
        image_degradation_group.setLayout(image_degradation_layout)
        left_layout.addWidget(image_degradation_group)

        self.add_technique_controls(image_degradation_layout, "Gaussian Noise", 
                                    [("Mean", 0, 50, 0, self.update_gaussian_mean),
                                     ("Sigma", 0, 100, 25, self.update_gaussian_sigma)])
        self.add_technique_controls(image_degradation_layout, "Salt and Pepper Noise", 
                                    [("Salt ", 0, 100, 2, self.update_salt_prob),
                                     ("Pepper ", 0, 100, 2, self.update_pepper_prob)])

        left_layout.addStretch(1)  # Add stretch to push controls to the top

        # Center: video display, face count, and chart
        center_layout = QVBoxLayout()
        main_layout.addLayout(center_layout, 4)  # Increased ratio for larger video display

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)  # Larger minimum size
        center_layout.addWidget(self.video_label)

        self.count_label = QLabel("Number of faces detected: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        center_layout.addWidget(self.count_label)

        self.face_count_plot = FaceCountPlot(self, width=5, height=2)
        center_layout.addWidget(self.face_count_plot)

        button_layout = QHBoxLayout()
        button_styles = {
            "Start": ("background-color: #4CAF50; color: white;", self.start_video),
            "Stop": ("background-color: #F44336; color: white;", self.stop_video),
            "Quit": ("background-color: #808080; color: white;", self.quit_application)
        }
        for button_name, (style, callback) in button_styles.items():
            button = QPushButton(button_name)
            button.setMinimumHeight(50)
            button.setStyleSheet(style)
            button.clicked.connect(callback)
            button_layout.addWidget(button)
        center_layout.addLayout(button_layout)

        # Right side: remaining controls
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 1)

        # 2. Group for Image Processing
        image_processing_group = QGroupBox("Step 1: Image Processing")
        image_processing_layout = QVBoxLayout()
        image_processing_group.setLayout(image_processing_layout)
        right_layout.addWidget(image_processing_group)

        self.add_technique_controls(image_processing_layout, "Gaussian Blur", 
                                    [("Kernel Size", 1, 21, 5, self.update_blur_kernel)])
        self.add_technique_controls(image_processing_layout, "Median Filter", 
                                    [("Kernel Size", 1, 21, 3, self.update_median_kernel)])
        self.add_technique_controls(image_processing_layout, "CLAHE", 
                                    [("Clip Limit", 1, 50, 20, self.update_clahe_clip),
                                     ("Tile Size", 1, 16, 8, self.update_clahe_tile)])
        self.add_technique_controls(image_processing_layout, "Unsharp Masking", 
                                    [("Kernel Size", 1, 21, 5, self.update_unsharp_kernel),
                                     ("Weight", 1, 50, 15, self.update_unsharp_weight)])

        # 3. Group for Segmentation
        segmentation_group = QGroupBox("Step 2: Segmentation")
        segmentation_layout = QVBoxLayout()
        segmentation_group.setLayout(segmentation_layout)
        right_layout.addWidget(segmentation_group)

        self.add_technique_controls(segmentation_layout, "Semantic Segmentation", [])

        # 4. Group for Skin Tone Segmentation
        skin_tone_group = QGroupBox("Step 3: Feature Extraction")
        skin_tone_layout = QVBoxLayout()
        skin_tone_group.setLayout(skin_tone_layout)
        right_layout.addWidget(skin_tone_group)

        self.add_technique_controls(skin_tone_layout, "Skin Tone (Colour Cropping)", [])

        # Increase font size for all widgets
        self.increase_font_size()

        # Set up video capture and models
        self.cap = cv2.VideoCapture(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).to(self.device)
        self.deeplabv3.eval()
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        # Preprocessing parameters
        self.blur_kernel = 5
        self.median_kernel = 3
        self.clahe_clip = 2.0
        self.clahe_tile = 8
        self.unsharp_kernel = 5
        self.unsharp_weight = 1.5
        self.gaussian_mean = 0
        self.gaussian_sigma = 25
        self.salt_prob = 0.02
        self.pepper_prob = 0.02

        # Set up timer for video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_input(self):
        if self.webcam_radio.isChecked():
            self.start_webcam()
        elif self.clipboard_radio.isChecked():
            self.get_clipboard_image()
        elif self.local_file_radio.isChecked():
            self.get_local_file_image()

    def get_local_file_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.current_frame = img
                self.process_single_image(img)
            else:
                print("Failed to load the image.")
        else:
            print("No file selected.")

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)  # 30 ms == 33.33 fps

    def get_clipboard_image(self):
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()
        if pixmap and not pixmap.isNull():
            image = pixmap.toImage()
            buffer = image.bits().asstring(image.byteCount())
            img = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert to BGR for OpenCV
            self.current_frame = img  # Store the current frame
            self.process_single_image(img)
        else:
            print("No image in clipboard")

    def process_single_image(self, image):
        processed_frame, face_count = self.process_frame(image)
        self.display_frame(processed_frame)
        self.count_label.setText(f"Number of faces detected: {face_count}")
        self.face_count_plot.update_plot(face_count)
        
        # Start a timer to continuously update the plot
        self.timer.start(500)  # Update every second

    def add_technique_controls(self, layout, technique_name, controls):
        group_box = QGroupBox(technique_name)
        group_layout = QVBoxLayout()
        
        checkbox = QCheckBox("Enable")
        checkbox.setChecked(False)  # Set to False to disable by default
        group_layout.addWidget(checkbox)
        
        for control_name, min_val, max_val, default_val, callback in controls:
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel(f"{control_name}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider.valueChanged.connect(callback)
            
            value_label = QLabel(f"{default_val}")
            slider.valueChanged.connect(lambda v, label=value_label: label.setText(f"{v}"))
            
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            
            group_layout.addLayout(slider_layout)
            
            checkbox.stateChanged.connect(slider.setEnabled)
            checkbox.stateChanged.connect(value_label.setEnabled)
            
            slider.setEnabled(False)
            value_label.setEnabled(False)
        
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        
        checkbox_name = technique_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '') + "_checkbox"
        setattr(self, checkbox_name, checkbox)

    def increase_font_size(self):
        font = QFont()
        font.setPointSize(12)  # Increase font size
        QApplication.setFont(font)
        for widget in self.findChildren((QLabel, QPushButton, QCheckBox, QGroupBox)):
            widget.setFont(font)

    def update_blur_kernel(self, value):
        self.blur_kernel = value if value % 2 == 1 else value + 1

    def update_clahe_clip(self, value):
        self.clahe_clip = value / 10.0

    def update_clahe_tile(self, value):
        self.clahe_tile = value

    def update_unsharp_kernel(self, value):
        self.unsharp_kernel = value if value % 2 == 1 else value + 1

    def update_unsharp_weight(self, value):
        self.unsharp_weight = value / 10.0

    def update_gaussian_mean(self, value):
        self.gaussian_mean = value

    def update_gaussian_sigma(self, value):
        self.gaussian_sigma = value

    def update_salt_prob(self, value):
        self.salt_prob = value / 1000.0  # Convert to probability

    def update_pepper_prob(self, value):
        self.pepper_prob = value / 1000.0  # Convert to probability

    def update_median_kernel(self, value):
        self.median_kernel = value if value % 2 == 1 else value + 1

    def apply_median_filter(self, image):
        return cv2.medianBlur(image, self.median_kernel)

    def start_video(self):
        self.face_count_plot.reset()
        self.start_input()

    def stop_video(self):
        self.timer.stop()
        if hasattr(self, 'cap'):
            self.cap.release()

    def update_frame(self):
        if self.webcam_radio.isChecked():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.process_single_image(frame)
        elif (self.clipboard_radio.isChecked() or self.local_file_radio.isChecked()) and self.current_frame is not None:
            # Re-process the current frame with any updated settings
            self.process_single_image(self.current_frame)
    
    def update_frame_rate(self, value):
        self.frame_interval = value
        self.frame_rate_label.setText(f"{value}")
        if self.timer.isActive():
            self.timer.start(self.frame_interval)

    def process_frame(self, frame):
        # Create a copy of the frame to work on
        processed_frame = frame.copy()

        # Apply image degradation techniques
        if self.gaussian_noise_checkbox.isChecked():
            processed_frame = apply_gaussian_noise(processed_frame, self.gaussian_mean, self.gaussian_sigma)

        if self.salt_and_pepper_noise_checkbox.isChecked():
            processed_frame = apply_salt_and_pepper_noise(processed_frame, self.salt_prob, self.pepper_prob)

        # Apply image processing techniques
        if self.gaussian_blur_checkbox.isChecked():
            processed_frame = cv2.GaussianBlur(processed_frame, (self.blur_kernel, self.blur_kernel), 0)

        if self.median_filter_checkbox.isChecked():
            processed_frame = self.apply_median_filter(processed_frame)
        
        if self.clahe_checkbox.isChecked():
            lab = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))
            cl = clahe.apply(l)
            processed_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        
        if self.unsharp_masking_checkbox.isChecked():
            gaussian = cv2.GaussianBlur(processed_frame, (self.unsharp_kernel, self.unsharp_kernel), 0)
            processed_frame = cv2.addWeighted(processed_frame, 1 + self.unsharp_weight, gaussian, -self.unsharp_weight, 0)
        
        if self.semantic_segmentation_checkbox.isChecked():
            processed_frame = self.segment_frame(processed_frame)
        
        if self.skin_tone_colour_cropping_checkbox.isChecked():
            processed_frame, _ = self.skin_tone_segmentation(processed_frame)
        
        # Detect faces
        boxes, face_count = self.detect_faces(processed_frame)
        
        # Draw bounding boxes on a separate copy of the processed frame
        final_frame = processed_frame.copy()
        self.draw_face_boxes(final_frame, boxes)
        
        return final_frame, face_count
    
    def detect_faces(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            face_count = len(boxes)
        else:
            face_count = 0
            boxes = []
        return boxes, face_count

    def draw_face_boxes(self, frame, boxes):
        for i, box in enumerate(boxes):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i+1}", (int(box[0]), int(box[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    def skin_tone_segmentation(self, image):
        # Convert the image from BGR to YCrCb
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Define multiple ranges for different skin tones
        skin_ranges = [
            (np.array([0, 135, 85], dtype=np.uint8), np.array([255, 180, 135], dtype=np.uint8)),  # Broader range
            (np.array([0, 140, 95], dtype=np.uint8), np.array([255, 185, 130], dtype=np.uint8))   # More specific range
        ]
        # Create a combined mask
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lower_skin, upper_skin in skin_ranges:
            mask = cv2.inRange(ycrcb_image, lower_skin, upper_skin)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        # Apply morphological operations to refine the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(combined_mask, kernel, iterations=1)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=3)
        # Close holes in the mask
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
        # Remove small blobs
        min_blob_size = 500  # Adjust this value as needed
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(skin_mask, connectivity=8)
        sizes = stats[:, -1]
        cleaned_mask = np.zeros(output.shape, dtype=np.uint8)
        for i in range(1, nb_components):
            if sizes[i] >= min_blob_size:
                cleaned_mask[output == i] = 255
        # Final dilation to slightly expand the remaining areas
        cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=4)
        # Apply the mask to the original image
        skin_segmented = cv2.bitwise_and(image, image, mask=cleaned_mask)
        return skin_segmented, cleaned_mask

    def segment_frame(self, frame):
        input_tensor = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.deeplabv3(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        mask = output_predictions == 15  # Assuming class 15 is 'person'
        frame[~mask] = [255, 255, 255]  # Set background to white
        return frame

    def detect_and_count_faces(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is not None:
            for i, box in enumerate(boxes):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"Face {i+1}", (int(box[0]), int(box[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            count = len(boxes)
        else:
            count = 0
        return frame, count

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def quit_application(self):
        self.cap.release()  # Release the camera
        QApplication.quit()  # Quit the application

    def closeEvent(self, event):
        self.cap.release()  # Ensure camera is released when closing the window
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceDetectionApp()
    window.show()
    sys.exit(app.exec_())