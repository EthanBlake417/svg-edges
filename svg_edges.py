import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                               QHBoxLayout, QWidget, QPushButton, QFileDialog,
                               QGraphicsView, QGraphicsScene, QSlider, QScrollArea,
                               QGridLayout, QGroupBox)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QMouseEvent
from PySide6.QtSvg import QSvgGenerator


class ImageLabel(QLabel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.last_position = None
        self.erasing = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = event.mimeData().urls()
        if files:
            file_path = files[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.main_window.load_image(file_path)

    def mousePressEvent(self, event: QMouseEvent):
        if self.main_window.edge_image is not None and not self.main_window.showing_original:
            # Start erasing
            self.erasing = True
            self.apply_eraser(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.erasing = False
        self.last_position = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.erasing and self.main_window.edge_image is not None and not self.main_window.showing_original:
            self.apply_eraser(event)

    def apply_eraser(self, event: QMouseEvent):
        # Convert event coordinates to image coordinates
        pixmap = self.pixmap()
        if pixmap and self.main_window.edge_image is not None:
            # Get image dimensions
            img_height, img_width = self.main_window.edge_image.shape[:2]

            # Get the actual pixmap size as displayed
            label_width = self.width()
            label_height = self.height()

            # Calculate scaling factor to maintain aspect ratio
            scale_x = label_width / img_width
            scale_y = label_height / img_height
            scale = min(scale_x, scale_y)

            # Calculate the displayed image size
            display_width = img_width * scale
            display_height = img_height * scale

            # Calculate offsets to center the image
            x_offset = (label_width - display_width) / 2
            y_offset = (label_height - display_height) / 2

            # Get mouse position relative to the QLabel
            mouse_x = event.position().x()
            mouse_y = event.position().y()

            # Check if mouse is within the image area
            if (x_offset <= mouse_x <= x_offset + display_width and
                    y_offset <= mouse_y <= y_offset + display_height):
                # Convert mouse coordinates to original image coordinates
                x = int((mouse_x - x_offset) / scale)
                y = int((mouse_y - y_offset) / scale)

                # Ensure coordinates are within bounds
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))

                # Call eraser method with the adjusted coordinates
                current_position = (x, y)
                self.main_window.apply_eraser(current_position)
                self.last_position = current_position


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection & SVG Converter")
        self.setMinimumSize(1200, 800)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side controls panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(400)

        # File controls group
        file_group = QGroupBox("File Controls")
        file_layout = QVBoxLayout()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(lambda: self.load_image())
        file_layout.addWidget(load_btn)

        save_btn = QPushButton("Save as SVG")
        save_btn.clicked.connect(self.save_svg)
        file_layout.addWidget(save_btn)

        self.toggle_btn = QPushButton("Toggle Original/Edge View")
        self.toggle_btn.clicked.connect(self.toggle_view)
        self.toggle_btn.setEnabled(False)
        file_layout.addWidget(self.toggle_btn)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # Eraser tool settings
        eraser_group = QGroupBox("Eraser Tool")
        eraser_layout = QVBoxLayout()

        eraser_explanation = QLabel(
            "Eraser Tool:\nClick and drag to erase black lines.\n"
            "Adjust eraser size to control precision."
        )
        eraser_explanation.setWordWrap(True)
        eraser_layout.addWidget(eraser_explanation)

        eraser_size_layout = QHBoxLayout()
        eraser_size_layout.addWidget(QLabel("Eraser Size:"))
        self.eraser_size = QSlider(Qt.Horizontal)
        self.eraser_size.setRange(1, 50)
        self.eraser_size.setValue(10)
        eraser_size_layout.addWidget(self.eraser_size)
        self.eraser_size_label = QLabel("10")
        self.eraser_size.valueChanged.connect(
            lambda v: self.eraser_size_label.setText(str(v)))
        eraser_size_layout.addWidget(self.eraser_size_label)
        eraser_layout.addLayout(eraser_size_layout)

        eraser_group.setLayout(eraser_layout)
        left_layout.addWidget(eraser_group)

        # Parameters group
        params_group = QGroupBox("Threshold Parameters")
        params_layout = QVBoxLayout()

        # Pre-processing smoothing
        pre_smooth_group = QGroupBox("Pre-processing Smoothing")
        pre_smooth_layout = QGridLayout()

        pre_smooth_explanation = QLabel(
            "Pre-processing Smoothing:\nApplies Gaussian blur before threshold.\n"
            "Higher values reduce noise but may lose detail."
        )
        pre_smooth_explanation.setWordWrap(True)
        pre_smooth_layout.addWidget(pre_smooth_explanation, 0, 0)

        self.pre_smooth = QSlider(Qt.Horizontal)
        self.pre_smooth.setRange(1, 21)
        self.pre_smooth.setValue(1)
        self.pre_smooth.valueChanged.connect(self.update_edge_detection)
        pre_smooth_layout.addWidget(self.pre_smooth, 1, 0)

        self.pre_smooth_label = QLabel("1")
        self.pre_smooth.valueChanged.connect(
            lambda v: self.pre_smooth_label.setText(str(v)))
        pre_smooth_layout.addWidget(self.pre_smooth_label, 1, 1)

        pre_smooth_group.setLayout(pre_smooth_layout)
        params_layout.addWidget(pre_smooth_group)

        # Threshold value
        threshold_group = QGroupBox("Threshold")
        threshold_layout = QGridLayout()

        threshold_explanation = QLabel(
            "Threshold Value:\nPixels above this value will be white,\n"
            "pixels below will be black."
        )
        threshold_explanation.setWordWrap(True)
        threshold_layout.addWidget(threshold_explanation, 0, 0)

        self.threshold = QSlider(Qt.Horizontal)
        self.threshold.setRange(0, 255)
        self.threshold.setValue(128)
        self.threshold.valueChanged.connect(self.update_edge_detection)
        threshold_layout.addWidget(self.threshold, 1, 0)

        self.threshold_label = QLabel("128")
        self.threshold.valueChanged.connect(
            lambda v: self.threshold_label.setText(str(v)))
        threshold_layout.addWidget(self.threshold_label, 1, 1)

        threshold_group.setLayout(threshold_layout)
        params_layout.addWidget(threshold_group)

        # Post-processing smoothing
        post_smooth_group = QGroupBox("Post-processing Smoothing")
        post_smooth_layout = QGridLayout()

        post_smooth_explanation = QLabel(
            "Post-processing Smoothing:\nApplies Gaussian blur after threshold.\n"
            "Helps smooth jagged edges."
        )
        post_smooth_explanation.setWordWrap(True)
        post_smooth_layout.addWidget(post_smooth_explanation, 0, 0)

        self.post_smooth = QSlider(Qt.Horizontal)
        self.post_smooth.setRange(1, 21)
        self.post_smooth.setValue(1)
        self.post_smooth.valueChanged.connect(self.update_edge_detection)
        post_smooth_layout.addWidget(self.post_smooth, 1, 0)

        self.post_smooth_label = QLabel("1")
        self.post_smooth.valueChanged.connect(
            lambda v: self.post_smooth_label.setText(str(v)))
        post_smooth_layout.addWidget(self.post_smooth_label, 1, 1)

        post_smooth_group.setLayout(post_smooth_layout)
        params_layout.addWidget(post_smooth_group)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        left_layout.addStretch()

        # Add left panel to main layout
        main_layout.addWidget(left_panel)

        # Right side image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Image display
        self.image_label = ImageLabel(self)
        self.image_label.setMinimumSize(400, 400)

        # Add image label to scroll area
        scroll_area.setWidget(self.image_label)
        right_layout.addWidget(scroll_area)

        # Add right panel to main layout
        main_layout.addWidget(right_panel)

        self.original_image = None
        self.edge_image = None
        self.showing_original = True

    def load_image(self, file_name=None):
        if file_name is None:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File",
                                                       "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_name:
            self.original_image = cv2.imread(file_name)
            if self.original_image is not None:
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                self.toggle_btn.setEnabled(False)
                self.showing_original = True
                self.update_edge_detection()

    def update_edge_detection(self):
        if self.original_image is not None:
            # Get pre-processing smoothness value (ensure it's odd)
            pre_smooth_val = self.pre_smooth.value()
            if pre_smooth_val % 2 == 0:
                pre_smooth_val += 1

            # Apply pre-processing smoothing if value > 1
            if pre_smooth_val > 1:
                processed = cv2.GaussianBlur(self.original_image, (pre_smooth_val, pre_smooth_val), 0)
            else:
                processed = self.original_image.copy()

            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Invert the grayscale image
            inverted = cv2.bitwise_not(gray)

            # Apply threshold to create binary image
            _, binary = cv2.threshold(inverted,
                                      self.threshold.value(),
                                      255,
                                      cv2.THRESH_BINARY)

            # Morphological closing to fill holes
            kernel = np.ones((60, 60), np.uint8)
            closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Find edges with Canny
            edges = cv2.Canny(closing, 100, 200)

            # Convert back to expected format (black objects on white background)
            edges = cv2.bitwise_not(edges)

            # Get post-processing smoothness value (ensure it's odd)
            post_smooth_val = self.post_smooth.value()
            if post_smooth_val % 2 == 0:
                post_smooth_val += 1

            # Apply post-processing smoothing if value > 1
            if post_smooth_val > 1:
                edges = cv2.GaussianBlur(edges, (post_smooth_val, post_smooth_val), 0)
                # Threshold again to make it binary after blur
                _, edges = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)

            # Convert edges to RGB format
            self.edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Update display if showing edges
            if not self.showing_original:
                self.display_edge_image()

            # Enable toggle button
            self.toggle_btn.setEnabled(True)

    def apply_eraser(self, current_pos):
        """Erase black lines by making them white at the clicked position"""
        if self.edge_image is None:
            return

        # Get eraser radius
        radius = self.eraser_size.value()

        # Convert to grayscale for processing
        gray = cv2.cvtColor(self.edge_image, cv2.COLOR_RGB2GRAY)

        # Draw white circle at the current position
        cv2.circle(gray, current_pos, radius, 255, -1)  # -1 means filled circle

        # Convert back to RGB
        self.edge_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Update display
        self.display_edge_image()

    def display_edge_image(self):
        if self.edge_image is not None:
            h, w, ch = self.edge_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.edge_image.copy().data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def toggle_view(self):
        if self.original_image is not None and self.edge_image is not None:
            if self.showing_original:
                self.display_edge_image()
                self.showing_original = False
            else:
                # Show original image
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                self.showing_original = True

    def save_svg(self):
        if self.edge_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save SVG File",
                                                       "", "SVG files (*.svg)")
            if file_name:
                # Create SVG generator
                generator = QSvgGenerator()
                generator.setFileName(file_name)
                generator.setSize(self.image_label.pixmap().size())
                generator.setViewBox(self.image_label.pixmap().rect())

                # Create painter and draw
                painter = QPainter()
                painter.begin(generator)
                painter.drawPixmap(0, 0, self.image_label.pixmap())
                painter.end()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())