import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                               QHBoxLayout, QWidget, QPushButton, QFileDialog,
                               QGraphicsView, QGraphicsScene, QSlider, QScrollArea,
                               QGridLayout, QGroupBox, QRadioButton, QButtonGroup,
                               QStackedWidget)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtSvg import QSvgGenerator


class ImageLabel(QLabel):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)

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

        # Edge detection mode group
        mode_group = QGroupBox("Edge Detection Mode")
        mode_layout = QVBoxLayout()

        self.mode_group = QButtonGroup()

        self.canny_radio = QRadioButton("Canny Edge Detection")
        self.canny_radio.setChecked(True)
        self.canny_radio.toggled.connect(self.update_edge_detection)
        self.mode_group.addButton(self.canny_radio)
        mode_layout.addWidget(self.canny_radio)

        self.threshold_radio = QRadioButton("Simple Threshold")
        self.threshold_radio.toggled.connect(self.update_edge_detection)
        self.mode_group.addButton(self.threshold_radio)
        mode_layout.addWidget(self.threshold_radio)

        self.laser_radio = QRadioButton("Laser Cutter")
        self.laser_radio.toggled.connect(self.update_edge_detection)
        self.mode_group.addButton(self.laser_radio)
        mode_layout.addWidget(self.laser_radio)

        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)

        # Parameters stack
        self.params_stack = QStackedWidget()

        # Canny parameters widget
        canny_widget = QWidget()
        canny_layout = QVBoxLayout(canny_widget)

        # Canny smoothing
        canny_smooth_group = QGroupBox("Smoothing")
        canny_smooth_layout = QGridLayout()

        canny_smooth_explanation = QLabel(
            "Smoothing:\nApplies Gaussian blur before detection.\n"
            "Higher values reduce noise but may lose detail."
        )
        canny_smooth_explanation.setWordWrap(True)
        canny_smooth_layout.addWidget(canny_smooth_explanation, 0, 0)

        self.canny_smooth = QSlider(Qt.Horizontal)
        self.canny_smooth.setRange(1, 21)
        self.canny_smooth.setValue(1)
        self.canny_smooth.valueChanged.connect(self.update_edge_detection)
        canny_smooth_layout.addWidget(self.canny_smooth, 1, 0)

        self.canny_smooth_label = QLabel("1")
        self.canny_smooth.valueChanged.connect(
            lambda v: self.canny_smooth_label.setText(str(v)))
        canny_smooth_layout.addWidget(self.canny_smooth_label, 1, 1)

        canny_smooth_group.setLayout(canny_smooth_layout)
        canny_layout.addWidget(canny_smooth_group)

        # Canny thresholds
        canny_threshold_group = QGroupBox("Thresholds")
        canny_threshold_layout = QGridLayout()

        threshold1_explanation = QLabel(
            "Lower Threshold:\nControls edge sensitivity.\n"
            "Lower values detect more edges but increase noise."
        )
        threshold1_explanation.setWordWrap(True)
        canny_threshold_layout.addWidget(threshold1_explanation, 0, 0)

        self.threshold1 = QSlider(Qt.Horizontal)
        self.threshold1.setRange(0, 255)
        self.threshold1.setValue(100)
        self.threshold1.valueChanged.connect(self.update_edge_detection)
        canny_threshold_layout.addWidget(self.threshold1, 1, 0)

        self.threshold1_label = QLabel("100")
        self.threshold1.valueChanged.connect(
            lambda v: self.threshold1_label.setText(str(v)))
        canny_threshold_layout.addWidget(self.threshold1_label, 1, 1)

        threshold2_explanation = QLabel(
            "Upper Threshold:\nControls edge continuity.\n"
            "Higher values create more continuous edges."
        )
        threshold2_explanation.setWordWrap(True)
        canny_threshold_layout.addWidget(threshold2_explanation, 2, 0)

        self.threshold2 = QSlider(Qt.Horizontal)
        self.threshold2.setRange(0, 255)
        self.threshold2.setValue(200)
        self.threshold2.valueChanged.connect(self.update_edge_detection)
        canny_threshold_layout.addWidget(self.threshold2, 3, 0)

        self.threshold2_label = QLabel("200")
        self.threshold2.valueChanged.connect(
            lambda v: self.threshold2_label.setText(str(v)))
        canny_threshold_layout.addWidget(self.threshold2_label, 3, 1)

        canny_threshold_group.setLayout(canny_threshold_layout)
        canny_layout.addWidget(canny_threshold_group)
        # Simple threshold parameters widget
        # Simple threshold parameters widget
        threshold_widget = QWidget()
        threshold_layout = QVBoxLayout(threshold_widget)

        # Simple threshold smoothing
        threshold_smooth_group = QGroupBox("Smoothing")
        threshold_smooth_layout = QGridLayout()

        threshold_smooth_explanation = QLabel(
            "Smoothing:\nApplies Gaussian blur before threshold.\n"
            "Higher values reduce noise but may lose detail."
        )
        threshold_smooth_explanation.setWordWrap(True)
        threshold_smooth_layout.addWidget(threshold_smooth_explanation, 0, 0)

        self.threshold_smooth = QSlider(Qt.Horizontal)
        self.threshold_smooth.setRange(1, 21)
        self.threshold_smooth.setValue(1)
        self.threshold_smooth.valueChanged.connect(self.update_edge_detection)
        threshold_smooth_layout.addWidget(self.threshold_smooth, 1, 0)

        self.threshold_smooth_label = QLabel("1")
        self.threshold_smooth.valueChanged.connect(
            lambda v: self.threshold_smooth_label.setText(str(v)))
        threshold_smooth_layout.addWidget(self.threshold_smooth_label, 1, 1)

        threshold_smooth_group.setLayout(threshold_smooth_layout)
        threshold_layout.addWidget(threshold_smooth_group)

        # Simple threshold value
        threshold_value_group = QGroupBox("Threshold")
        threshold_value_layout = QGridLayout()

        threshold_explanation = QLabel(
            "Threshold Value:\nPixels above this value will be white,\n"
            "pixels below will be black."
        )
        threshold_explanation.setWordWrap(True)
        threshold_value_layout.addWidget(threshold_explanation, 0, 0)

        self.simple_threshold = QSlider(Qt.Horizontal)
        self.simple_threshold.setRange(0, 255)
        self.simple_threshold.setValue(128)
        self.simple_threshold.valueChanged.connect(self.update_edge_detection)
        threshold_value_layout.addWidget(self.simple_threshold, 1, 0)

        self.simple_threshold_label = QLabel("128")
        self.simple_threshold.valueChanged.connect(
            lambda v: self.simple_threshold_label.setText(str(v)))
        threshold_value_layout.addWidget(self.simple_threshold_label, 1, 1)

        threshold_value_group.setLayout(threshold_value_layout)
        threshold_layout.addWidget(threshold_value_group)

        # Laser Cutter parameters widget
        laser_widget = QWidget()
        laser_layout = QVBoxLayout(laser_widget)

        # Laser Cutter smoothing
        laser_smooth_group = QGroupBox("Smoothing")
        laser_smooth_layout = QGridLayout()

        laser_smooth_explanation = QLabel(
            "Smoothing:\nApplies Gaussian blur before detection.\n"
            "Higher values reduce noise but may lose detail."
        )
        laser_smooth_explanation.setWordWrap(True)
        laser_smooth_layout.addWidget(laser_smooth_explanation, 0, 0)

        self.laser_smooth = QSlider(Qt.Horizontal)
        self.laser_smooth.setRange(1, 21)
        self.laser_smooth.setValue(1)
        self.laser_smooth.valueChanged.connect(self.update_edge_detection)
        laser_smooth_layout.addWidget(self.laser_smooth, 1, 0)

        self.laser_smooth_label = QLabel("1")
        self.laser_smooth.valueChanged.connect(
            lambda v: self.laser_smooth_label.setText(str(v)))
        laser_smooth_layout.addWidget(self.laser_smooth_label, 1, 1)

        laser_smooth_group.setLayout(laser_smooth_layout)
        laser_layout.addWidget(laser_smooth_group)

        # Laser Cutter threshold
        laser_threshold_group = QGroupBox("Threshold")
        laser_threshold_layout = QGridLayout()

        laser_threshold_explanation = QLabel(
            "Threshold Value:\nControls the sensitivity of edge detection.\n"
            "Lower values detect more edges."
        )
        laser_threshold_explanation.setWordWrap(True)
        laser_threshold_layout.addWidget(laser_threshold_explanation, 0, 0)

        self.laser_threshold = QSlider(Qt.Horizontal)
        self.laser_threshold.setRange(0, 255)
        self.laser_threshold.setValue(128)
        self.laser_threshold.valueChanged.connect(self.update_edge_detection)
        laser_threshold_layout.addWidget(self.laser_threshold, 1, 0)

        self.laser_threshold_label = QLabel("128")
        self.laser_threshold.valueChanged.connect(
            lambda v: self.laser_threshold_label.setText(str(v)))
        laser_threshold_layout.addWidget(self.laser_threshold_label, 1, 1)

        laser_threshold_group.setLayout(laser_threshold_layout)
        laser_layout.addWidget(laser_threshold_group)

        # Add widgets to stack
        self.params_stack.addWidget(canny_widget)
        self.params_stack.addWidget(threshold_widget)
        self.params_stack.addWidget(laser_widget)

        # Connect radio buttons to stack
        self.canny_radio.toggled.connect(lambda: self.params_stack.setCurrentIndex(0))
        self.threshold_radio.toggled.connect(lambda: self.params_stack.setCurrentIndex(1))
        self.laser_radio.toggled.connect(lambda: self.params_stack.setCurrentIndex(2))

        left_layout.addWidget(self.params_stack)
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
            # Get current smoothness value (ensure it's odd)
            if self.canny_radio.isChecked():
                smooth_val = self.canny_smooth.value()
            elif self.threshold_radio.isChecked():
                smooth_val = self.threshold_smooth.value()
            else:  # Laser Cutter mode
                smooth_val = self.laser_smooth.value()

            if smooth_val % 2 == 0:
                smooth_val += 1

            # Apply smoothing if value > 1
            if smooth_val > 1:
                processed = cv2.GaussianBlur(self.original_image, (smooth_val, smooth_val), 0)
            else:
                processed = self.original_image.copy()

            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            if self.canny_radio.isChecked():
                # Apply Canny edge detection
                edges = cv2.Canny(gray,
                                  self.threshold1.value(),
                                  self.threshold2.value())
            elif self.threshold_radio.isChecked():
                # Invert the grayscale image
                inverted = cv2.bitwise_not(gray)

                # Apply threshold
                _, binary = cv2.threshold(inverted,
                                          self.simple_threshold.value(),
                                          255,
                                          cv2.THRESH_BINARY)

                # Morphological closing to fill holes
                kernel = np.ones((15, 15), np.uint8)  # you may need to adjust kernel size
                closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # Find edges with Canny
                edges = cv2.Canny(closing, 100, 200)

                # Optional: Convert back to expected format (black objects on white background)
                edges = cv2.bitwise_not(edges)
            # elif self.threshold_radio.isChecked():
            #     # Apply simple threshold
            #     _, edges = cv2.threshold(gray,
            #                              self.simple_threshold.value(),
            #                              255,
            #                              cv2.THRESH_BINARY)  # white background, black objects
            #
            #     # Find connected components
            #     # The first return value is number of labels, second is the label matrix
            #     num_labels, labels = cv2.connectedComponents(cv2.bitwise_not(edges))  # invert because connectedComponents looks for white regions
            #
            #     # Create output image (white background)
            #     edges = np.ones_like(edges) * 255
            #
            #     # Fill each labeled region in black
            #     for label in range(1, num_labels):  # start from 1 to skip background
            #         edges[labels == label] = 0
            # elif self.threshold_radio.isChecked():
            #     # Apply simple threshold
            #     _, edges = cv2.threshold(gray,
            #                              self.simple_threshold.value(),
            #                              255,
            #                              cv2.THRESH_BINARY)
            #
            #     # Replace black border with white by flood filling from corners
            #     mask = np.zeros((edges.shape[0] + 2, edges.shape[1] + 2), dtype=np.uint8)
            #     cv2.floodFill(edges, mask, (0, 0), 255)  # Top-left corner
            #     cv2.floodFill(edges, mask, (edges.shape[1] - 1, 0), 255)  # Top-right
            #     cv2.floodFill(edges, mask, (0, edges.shape[0] - 1), 255)  # Bottom-left
            #     cv2.floodFill(edges, mask, (edges.shape[1] - 1, edges.shape[0] - 1), 255)  # Bottom-right
            #
            #     # Create a copy before finding contours
            #     edges_copy = edges.copy()
            #
            #     # Find all individual objects (black regions)
            #     contours, _ = cv2.findContours(edges_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     print(contours)
            #
            #     # Optional: Filter contours by size if needed
            #     min_contour_area = 100  # Adjust this value as needed
            #     valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            #     # print(valid_contours)
            #
            #     # Fill each valid contour
            #     cv2.drawContours(edges, valid_contours, -1, 0, -1)  # Fill each object in black

                # # Find all individual objects (black regions)
                # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #
                # # Fill each object
                # cv2.drawContours(edges, contours, -1, 0, -1)  # Fill each object in black

                # # Find all individual objects (black regions)
                # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #
                # # Fill each object
                # cv2.drawContours(edges, contours, -1, 0, -1)  # Fill each object in black
                #
                # # Remove the border we added
                # edges = edges[20:-20, 20:-20]
            #     # Apply simple threshold
            #     _, edges = cv2.threshold(gray,
            #                              self.simple_threshold.value(),
            #                              255,
            #                              cv2.THRESH_BINARY)
            else:  # Laser Cutter mode
                # First apply simple threshold to get binary image
                _, binary = cv2.threshold(gray,
                                          self.laser_threshold.value(),
                                          255,
                                          cv2.THRESH_BINARY_INV)  # Invert so objects are white

                # Fill all holes (flood fill from borders)
                filled = binary.copy()
                mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
                cv2.floodFill(filled, mask, (0, 0), 0)  # Fill from corner

                # Invert the filled image to get solid objects
                filled_inv = cv2.bitwise_not(filled)

                # Combine with original to get all filled regions
                filled_all = binary | filled_inv

                # Find the external contours of the filled regions
                contours, _ = cv2.findContours(filled_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create final white image
                edges = np.ones_like(gray) * 255

                # Draw only external contours in black
                cv2.drawContours(edges, contours, -1, (0, 0, 0), 1)

            # Convert edges to RGB format
            self.edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            # Update display if showing edges
            if not self.showing_original:
                self.display_edge_image()

            # Enable toggle button
            self.toggle_btn.setEnabled(True)

    def display_edge_image(self):
        if self.edge_image is not None:
            h, w, ch = self.edge_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(self.edge_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
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