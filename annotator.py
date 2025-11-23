# Copyright (c) 2024. SAM3 Annotator Tool
# GUI-based annotation tool using SAM3 for text and box prompting

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# SAM3 imports
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Convert binary mask to polygon coordinates (COCO format)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            contour = contour.flatten().tolist()
            polygons.append(contour)
    return polygons


class Annotation:
    """Represents a single annotation (mask with label)."""

    def __init__(
        self,
        mask: np.ndarray,
        label: str,
        score: float = 1.0,
        color: Optional[Tuple[int, int, int]] = None
    ):
        self.mask = mask
        self.label = label
        self.score = score
        self.color = color or self._generate_color()
        self.visible = True
        self.selected = False

    def _generate_color(self) -> Tuple[int, int, int]:
        return (
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255)
        )

    def to_coco_annotation(self, annotation_id: int, image_id: int, category_id: int) -> Dict:
        """Convert to COCO annotation format."""
        polygons = mask_to_polygon(self.mask)
        if not polygons:
            return None

        area = float(np.sum(self.mask))
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]

        return {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": polygons,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0
        }


class ImageCanvas(QLabel):
    """Canvas widget for displaying images with box drawing and point prompt support."""

    box_drawn = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    mask_clicked = pyqtSignal(int, int)
    point_added = pyqtSignal()  # Signal when points change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")

        self.original_image: Optional[np.ndarray] = None
        self.display_pixmap: Optional[QPixmap] = None
        self.annotations: List[Annotation] = []
        self.temp_masks: List[np.ndarray] = []
        self.temp_scores: List[float] = []

        self.scale_factor = 1.0
        self.zoom_factor = 1.0  # User-controlled zoom

        # Box drawing mode
        self.box_mode = False
        self.edit_mode = False
        self.point_mode = False  # Point prompt mode
        self.drawing_box = False
        self.box_start: Optional[Tuple[int, int]] = None
        self.box_end: Optional[Tuple[int, int]] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2

        # Point prompt data (for multiple points)
        self.positive_points: List[Tuple[int, int]] = []  # Left click = positive
        self.negative_points: List[Tuple[int, int]] = []  # Right click = negative

        self.mask_opacity = 0.5

        # Cache for overlay
        self._cached_overlay: Optional[np.ndarray] = None
        self._cache_valid = False

    def set_image(self, image: np.ndarray):
        """Set the image to display."""
        self.original_image = image.copy()
        self.annotations = []
        self.temp_masks = []
        self.temp_scores = []
        self.current_box = None
        self.positive_points = []
        self.negative_points = []
        self.zoom_factor = 1.0
        self._cache_valid = False
        self._update_display()

    def clear_points(self):
        """Clear all prompt points."""
        self.positive_points = []
        self.negative_points = []
        self._cache_valid = False
        self._update_display()

    def has_points(self) -> bool:
        """Check if there are any prompt points."""
        return len(self.positive_points) > 0 or len(self.negative_points) > 0

    def set_temp_masks(self, masks: List[np.ndarray], scores: List[float]):
        """Set temporary masks from SAM3 prediction."""
        self.temp_masks = masks
        self.temp_scores = scores
        self._cache_valid = False
        self._update_display()

    def clear_temp_masks(self):
        """Clear temporary masks."""
        self.temp_masks = []
        self.temp_scores = []
        self._cache_valid = False
        self._update_display()

    def add_annotation(self, annotation: Annotation):
        """Add an annotation to the canvas."""
        self.annotations.append(annotation)
        self._cache_valid = False
        self._update_display()

    def remove_annotation(self, index: int):
        """Remove an annotation by index."""
        if 0 <= index < len(self.annotations):
            del self.annotations[index]
            self._cache_valid = False
            self._update_display()

    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations = []
        self._cache_valid = False
        self._update_display()

    def get_mask_at_point(self, x: int, y: int) -> int:
        """Get annotation index at given point."""
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            if ann.visible and 0 <= y < ann.mask.shape[0] and 0 <= x < ann.mask.shape[1]:
                if ann.mask[y, x]:
                    return i
        return -1

    def _update_display(self):
        """Update the display with optimized rendering."""
        if self.original_image is None:
            return

        # Start with original image
        display = self.original_image.copy()

        # Draw annotations efficiently using vectorized operations
        if self.annotations or self.temp_masks:
            # Create combined mask overlay
            overlay = np.zeros_like(display, dtype=np.float32)
            mask_combined = np.zeros(display.shape[:2], dtype=bool)

            # Draw confirmed annotations
            for ann in self.annotations:
                if ann.visible:
                    mask_bool = ann.mask.astype(bool)
                    color = np.array(ann.color[::-1], dtype=np.float32)  # BGR
                    overlay[mask_bool] = color
                    mask_combined |= mask_bool

                    # Draw contour
                    contours, _ = cv2.findContours(ann.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    border_color = (0, 255, 255) if ann.selected else ann.color[::-1]
                    cv2.drawContours(display, contours, -1, border_color, 2)

                    # Draw label
                    if contours:
                        M = cv2.moments(contours[0])
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(display, ann.label, (cx - 30, cy),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw temporary masks
            temp_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
            for i, (mask, score) in enumerate(zip(self.temp_masks, self.temp_scores)):
                mask_bool = mask.astype(bool)
                color = np.array(temp_colors[i % len(temp_colors)], dtype=np.float32)
                overlay[mask_bool] = color
                mask_combined |= mask_bool

                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, temp_colors[i % len(temp_colors)], 2)

                if contours:
                    M = cv2.moments(contours[0])
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(display, f"{i}: {score:.2f}", (cx - 20, cy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Apply overlay with alpha blending (vectorized)
            display = display.astype(np.float32)
            display[mask_combined] = (
                display[mask_combined] * (1 - self.mask_opacity) +
                overlay[mask_combined] * self.mask_opacity
            )
            display = display.astype(np.uint8)

        # Draw current box being drawn
        if self.current_box is not None:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw prompt points
        for px, py in self.positive_points:
            cv2.circle(display, (px, py), 8, (0, 255, 0), -1)  # Green = positive
            cv2.circle(display, (px, py), 8, (255, 255, 255), 2)  # White border
        for px, py in self.negative_points:
            cv2.circle(display, (px, py), 8, (255, 0, 0), -1)  # Red = negative
            cv2.circle(display, (px, py), 8, (255, 255, 255), 2)  # White border

        # Convert to QPixmap (image is already in RGB format from PIL)
        h, w = display.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_pixmap = QPixmap.fromImage(q_image)

        self._fit_to_widget()

    def _fit_to_widget(self):
        """Scale pixmap to fit widget."""
        if self.display_pixmap is None:
            return

        widget_size = self.size()
        pixmap_size = self.display_pixmap.size()

        scale_w = widget_size.width() / pixmap_size.width()
        scale_h = widget_size.height() / pixmap_size.height()
        base_scale = min(scale_w, scale_h, 1.0)
        self.scale_factor = base_scale * self.zoom_factor

        scaled_pixmap = self.display_pixmap.scaled(
            int(pixmap_size.width() * self.scale_factor),
            int(pixmap_size.height() * self.scale_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_widget()

    def _get_image_coords(self, pos) -> Tuple[int, int]:
        """Convert widget position to image coordinates."""
        pixmap = self.pixmap()
        if pixmap is None or self.original_image is None:
            return -1, -1

        label_rect = self.rect()
        pixmap_rect = pixmap.rect()
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2

        px = pos.x() - x_offset
        py = pos.y() - y_offset

        if px < 0 or py < 0 or px >= pixmap_rect.width() or py >= pixmap_rect.height():
            return -1, -1

        img_x = int(px / self.scale_factor)
        img_y = int(py / self.scale_factor)

        img_x = max(0, min(img_x, self.original_image.shape[1] - 1))
        img_y = max(0, min(img_y, self.original_image.shape[0] - 1))

        return img_x, img_y

    def mousePressEvent(self, event):
        if self.original_image is None:
            return

        img_x, img_y = self._get_image_coords(event.pos())
        if img_x < 0:
            return

        if self.box_mode and event.button() == Qt.LeftButton:
            self.drawing_box = True
            self.box_start = (img_x, img_y)
            self.box_end = (img_x, img_y)
            self.current_box = (img_x, img_y, img_x, img_y)
            self._update_display()
        elif self.point_mode:
            # Left click = positive point, Right click = negative point
            if event.button() == Qt.LeftButton:
                self.positive_points.append((img_x, img_y))
                self._cache_valid = False
                self._update_display()
                self.point_added.emit()
            elif event.button() == Qt.RightButton:
                self.negative_points.append((img_x, img_y))
                self._cache_valid = False
                self._update_display()
                self.point_added.emit()
        elif self.edit_mode and event.button() == Qt.LeftButton:
            self.mask_clicked.emit(img_x, img_y)

    def mouseMoveEvent(self, event):
        if self.drawing_box and self.box_start is not None:
            img_x, img_y = self._get_image_coords(event.pos())
            if img_x >= 0:
                self.box_end = (img_x, img_y)
                x1 = min(self.box_start[0], img_x)
                y1 = min(self.box_start[1], img_y)
                x2 = max(self.box_start[0], img_x)
                y2 = max(self.box_start[1], img_y)
                self.current_box = (x1, y1, x2, y2)
                self._update_display()

    def mouseReleaseEvent(self, event):
        if self.drawing_box and event.button() == Qt.LeftButton:
            self.drawing_box = False
            if self.box_start and self.box_end:
                x1 = min(self.box_start[0], self.box_end[0])
                y1 = min(self.box_start[1], self.box_end[1])
                x2 = max(self.box_start[0], self.box_end[0])
                y2 = max(self.box_start[1], self.box_end[1])
                # Only emit if box has meaningful size
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    self.box_drawn.emit(x1, y1, x2, y2)
            self.box_start = None
            self.box_end = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if self.original_image is None:
            return

        # Zoom in/out with mouse wheel
        delta = event.angleDelta().y()
        zoom_speed = 0.1
        
        if delta > 0:
            # Zoom in
            self.zoom_factor = min(self.zoom_factor + zoom_speed, 5.0)
        else:
            # Zoom out
            self.zoom_factor = max(self.zoom_factor - zoom_speed, 0.1)
        
        self._fit_to_widget()
        event.accept()


class OriginalImageCanvas(QLabel):
    """Canvas for displaying original image only."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        self.original_image: Optional[np.ndarray] = None
        self._pixmap_cache: Optional[QPixmap] = None
        self.zoom_factor = 1.0

    def set_image(self, image: np.ndarray):
        self.original_image = image.copy()
        self._pixmap_cache = None
        self.zoom_factor = 1.0
        self._update_display()

    def _update_display(self):
        if self.original_image is None:
            return

        if self._pixmap_cache is None:
            h, w = self.original_image.shape[:2]
            # Image is already in RGB format from PIL
            bytes_per_line = 3 * w
            q_image = QImage(self.original_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._pixmap_cache = QPixmap.fromImage(q_image)

        widget_size = self.size()
        pixmap_size = self._pixmap_cache.size()
        scale_w = widget_size.width() / pixmap_size.width()
        scale_h = widget_size.height() / pixmap_size.height()
        base_scale = min(scale_w, scale_h, 1.0)
        scale = base_scale * self.zoom_factor

        scaled_pixmap = self._pixmap_cache.scaled(
            int(pixmap_size.width() * scale),
            int(pixmap_size.height() * scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        if self.original_image is None:
            return

        # Zoom in/out with mouse wheel
        delta = event.angleDelta().y()
        zoom_speed = 0.1
        
        if delta > 0:
            # Zoom in
            self.zoom_factor = min(self.zoom_factor + zoom_speed, 5.0)
        else:
            # Zoom out
            self.zoom_factor = max(self.zoom_factor - zoom_speed, 0.1)
        
        self._update_display()
        event.accept()


class LabelDialog(QDialog):
    """Dialog for editing label or deleting annotation."""

    def __init__(self, current_label: str, existing_labels: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Annotation")
        self.setModal(True)
        self.result_action = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Current label: {current_label}"))

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("New label:"))
        self.label_input = QLineEdit(current_label)
        input_layout.addWidget(self.label_input)
        layout.addLayout(input_layout)

        if existing_labels:
            layout.addWidget(QLabel("Or select existing:"))
            self.label_combo = QComboBox()
            self.label_combo.addItems(existing_labels)
            self.label_combo.currentTextChanged.connect(self.label_input.setText)
            layout.addWidget(self.label_combo)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_save)
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("background-color: #cc4444;")
        delete_btn.clicked.connect(self._on_delete)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_save(self):
        self.result_action = 'edit'
        self.accept()

    def _on_delete(self):
        self.result_action = 'delete'
        self.accept()

    def get_label(self) -> str:
        return self.label_input.text().strip()


class SAM3Annotator(QMainWindow):
    """Main window for SAM3 annotation tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Annotator")
        self.setMinimumSize(1400, 800)

        self.model = None
        self.processor = None
        self.inference_state = None

        self.current_image_path: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.pil_image: Optional[Image.Image] = None

        self.image_list: List[str] = []
        self.current_image_index = 0

        self.categories: Dict[str, int] = {}
        self.next_category_id = 1

        self.output_folder: Optional[str] = None
        self.current_text_prompt: str = ""

        # Proposal storage (multiple masks from SAM3)
        self.proposal_masks: List[List[np.ndarray]] = []  # List of mask proposals
        self.proposal_scores: List[List[float]] = []
        self.selected_proposal_idx: int = 0

        self._setup_ui()
        self._setup_connections()
        self._setup_shortcuts()

        self.statusBar().showMessage("Ready. Load model and image. Shortcuts: S=save, A=accept, R=reject, D=delete, 1-4=proposals")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Image panel with two canvases
        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)

        # Original image (left)
        orig_container = QWidget()
        orig_layout = QVBoxLayout(orig_container)
        orig_layout.addWidget(QLabel("Original Image"))
        self.original_canvas = OriginalImageCanvas()
        orig_layout.addWidget(self.original_canvas, 1)
        image_layout.addWidget(orig_container)

        # Result image (right)
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.addWidget(QLabel("Result (Annotations)"))
        self.canvas = ImageCanvas()
        result_layout.addWidget(self.canvas, 1)
        image_layout.addWidget(result_container)

        main_splitter.addWidget(image_panel)

        # Right panel - Controls
        right_panel = QWidget()
        right_panel.setMaximumWidth(320)
        right_panel.setMinimumWidth(280)
        right_layout = QVBoxLayout(right_panel)

        # File controls
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        self.load_image_btn = QPushButton("Load Image")
        self.load_folder_btn = QPushButton("Load Folder")

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev (←)")
        self.next_btn = QPushButton("Next (→) >")
        self.image_info_label = QLabel("No image")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.image_info_label)
        nav_layout.addWidget(self.next_btn)

        self.output_folder_btn = QPushButton("Set Output Folder")
        self.output_folder_label = QLabel("Not set")
        self.output_folder_label.setStyleSheet("color: orange;")

        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.load_folder_btn)
        file_layout.addLayout(nav_layout)
        file_layout.addWidget(self.output_folder_btn)
        file_layout.addWidget(self.output_folder_label)

        right_layout.addWidget(file_group)

        # Model controls
        model_group = QGroupBox("SAM3 Model")
        model_layout = QVBoxLayout(model_group)

        self.load_model_btn = QPushButton("Load Model")
        self.model_status = QLabel("Model: Not loaded")
        self.model_status.setStyleSheet("color: red;")

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("0.50")
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)

        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_status)
        model_layout.addLayout(conf_layout)

        right_layout.addWidget(model_group)

        # Text prompt controls
        text_group = QGroupBox("Text Prompt")
        text_layout = QVBoxLayout(text_group)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., 'mushroom cap'")
        self.text_prompt_btn = QPushButton("Generate Masks")

        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.text_prompt_btn)

        right_layout.addWidget(text_group)

        # Box prompt controls
        box_group = QGroupBox("Box Prompt")
        box_layout = QVBoxLayout(box_group)

        self.box_mode_cb = QCheckBox("Enable Box Mode")
        box_info = QLabel("Drag on result image to draw box")
        box_info.setStyleSheet("color: gray; font-size: 10px;")

        self.box_prompt_btn = QPushButton("Generate from Box")
        self.clear_box_btn = QPushButton("Clear Box")

        box_layout.addWidget(self.box_mode_cb)
        box_layout.addWidget(box_info)
        box_layout.addWidget(self.box_prompt_btn)
        box_layout.addWidget(self.clear_box_btn)

        right_layout.addWidget(box_group)

        # Point prompt controls
        point_group = QGroupBox("Point Prompt")
        point_layout = QVBoxLayout(point_group)

        self.point_mode_cb = QCheckBox("Enable Point Mode")
        point_info = QLabel("Left=Positive(green), Right=Negative(red)")
        point_info.setStyleSheet("color: gray; font-size: 10px;")

        self.point_prompt_btn = QPushButton("Generate from Points")
        self.clear_points_btn = QPushButton("Clear Points (R)")
        self.point_status_label = QLabel("Points: 0 pos, 0 neg")

        point_layout.addWidget(self.point_mode_cb)
        point_layout.addWidget(point_info)
        point_layout.addWidget(self.point_status_label)
        point_layout.addWidget(self.point_prompt_btn)
        point_layout.addWidget(self.clear_points_btn)

        right_layout.addWidget(point_group)

        # Mask management with proposals
        mask_group = QGroupBox("Mask Proposals")
        mask_layout = QVBoxLayout(mask_group)

        # Proposal buttons (1, 2, 3, 4 shortcuts)
        proposal_label = QLabel("Select proposal (shortcuts 1-4):")
        proposal_label.setStyleSheet("color: gray; font-size: 10px;")
        mask_layout.addWidget(proposal_label)

        proposal_btn_layout = QHBoxLayout()
        self.proposal_btns = []
        for i in range(4):
            btn = QPushButton(f"{i+1}")
            btn.setMinimumHeight(40)
            btn.setToolTip(f"Select proposal {i+1} (shortcut: {i+1})")
            self.proposal_btns.append(btn)
            proposal_btn_layout.addWidget(btn)
        mask_layout.addLayout(proposal_btn_layout)

        mask_select_layout = QHBoxLayout()
        mask_select_layout.addWidget(QLabel("Select:"))
        self.mask_selector = QSpinBox()
        self.mask_selector.setMinimum(0)
        self.mask_selector.setMaximum(99)
        mask_select_layout.addWidget(self.mask_selector)

        self.accept_mask_btn = QPushButton("Accept Selected (A)")
        self.accept_all_btn = QPushButton("Accept All")
        self.clear_temp_btn = QPushButton("Clear Temp")

        mask_layout.addLayout(mask_select_layout)
        mask_layout.addWidget(self.accept_mask_btn)
        mask_layout.addWidget(self.accept_all_btn)
        mask_layout.addWidget(self.clear_temp_btn)

        right_layout.addWidget(mask_group)

        # Edit mode
        edit_group = QGroupBox("Edit Annotations")
        edit_layout = QVBoxLayout(edit_group)

        self.edit_mode_cb = QCheckBox("Edit Mode (click mask)")
        self.delete_selected_btn = QPushButton("Delete Selected (D)")
        self.clear_ann_btn = QPushButton("Clear All Annotations")

        edit_layout.addWidget(self.edit_mode_cb)
        edit_layout.addWidget(self.delete_selected_btn)
        edit_layout.addWidget(self.clear_ann_btn)

        right_layout.addWidget(edit_group)

        # Annotations list
        ann_group = QGroupBox("Annotations")
        ann_layout = QVBoxLayout(ann_group)
        self.ann_list = QListWidget()
        self.ann_list.setMaximumHeight(120)
        ann_layout.addWidget(self.ann_list)
        right_layout.addWidget(ann_group)

        # Display controls
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        opacity_layout.addWidget(self.opacity_slider)

        save_label = QLabel("Press 'S' to save")
        save_label.setStyleSheet("color: #88ff88; font-weight: bold;")

        display_layout.addLayout(opacity_layout)
        display_layout.addWidget(save_label)

        right_layout.addWidget(display_group)
        right_layout.addStretch()

        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1100, 300])

    def _setup_connections(self):
        self.load_image_btn.clicked.connect(self._load_image)
        self.load_folder_btn.clicked.connect(self._load_folder)
        self.output_folder_btn.clicked.connect(self._select_output_folder)
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn.clicked.connect(self._next_image)

        self.load_model_btn.clicked.connect(self._load_model)
        self.confidence_slider.valueChanged.connect(self._update_confidence)

        self.text_prompt_btn.clicked.connect(self._text_prompt)
        self.text_input.returnPressed.connect(self._text_prompt)

        self.box_mode_cb.toggled.connect(self._toggle_box_mode)
        self.box_prompt_btn.clicked.connect(self._box_prompt)
        self.clear_box_btn.clicked.connect(self._clear_box)
        self.canvas.box_drawn.connect(self._on_box_drawn)

        # Point prompt connections
        self.point_mode_cb.toggled.connect(self._toggle_point_mode)
        self.point_prompt_btn.clicked.connect(self._point_prompt)
        self.clear_points_btn.clicked.connect(self._clear_points)
        self.canvas.point_added.connect(self._on_point_added)

        # Proposal buttons
        for i, btn in enumerate(self.proposal_btns):
            btn.clicked.connect(lambda checked, idx=i: self._select_proposal(idx))

        self.accept_mask_btn.clicked.connect(self._accept_selected_mask)
        self.accept_all_btn.clicked.connect(self._accept_all_masks)
        self.clear_temp_btn.clicked.connect(self._clear_temp_masks)

        self.edit_mode_cb.toggled.connect(self._toggle_edit_mode)
        self.canvas.mask_clicked.connect(self._on_mask_clicked)
        self.delete_selected_btn.clicked.connect(self._delete_selected_annotation)
        self.clear_ann_btn.clicked.connect(self._clear_annotations)

        self.ann_list.itemDoubleClicked.connect(self._on_annotation_double_clicked)
        self.opacity_slider.valueChanged.connect(self._update_opacity)

    def _setup_shortcuts(self):
        # Navigation
        QShortcut(QKeySequence('Left'), self).activated.connect(self._prev_image)
        QShortcut(QKeySequence('Right'), self).activated.connect(self._next_image)

        # Save
        QShortcut(QKeySequence('S'), self).activated.connect(self._quick_save)

        # Accept/Reject
        QShortcut(QKeySequence('A'), self).activated.connect(self._accept_selected_mask)
        QShortcut(QKeySequence('R'), self).activated.connect(self._reject_all)

        # Delete
        QShortcut(QKeySequence('D'), self).activated.connect(self._delete_selected_annotation)

        # Proposal selection (1, 2, 3, 4)
        QShortcut(QKeySequence('1'), self).activated.connect(lambda: self._select_proposal(0))
        QShortcut(QKeySequence('2'), self).activated.connect(lambda: self._select_proposal(1))
        QShortcut(QKeySequence('3'), self).activated.connect(lambda: self._select_proposal(2))
        QShortcut(QKeySequence('4'), self).activated.connect(lambda: self._select_proposal(3))

    def _select_output_folder(self):
        """Select output folder - does NOT affect image list."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            display_path = folder if len(folder) < 30 else "..." + folder[-27:]
            self.output_folder_label.setText(display_path)
            self.output_folder_label.setStyleSheet("color: #88ff88;")
            self.output_folder_label.setToolTip(folder)
            self.statusBar().showMessage(f"Output folder: {folder}")

    def _quick_save(self):
        if not self.canvas.annotations:
            self.statusBar().showMessage("No annotations to save")
            return

        if self.output_folder is None:
            QMessageBox.warning(self, "Warning", "Please set output folder first")
            self._select_output_folder()
            if self.output_folder is None:
                return

        self._save_annotations_to_folder()

    def _save_annotations_to_folder(self):
        if self.current_image_path is None:
            self.statusBar().showMessage("No image loaded")
            return

        try:
            image_name = os.path.basename(self.current_image_path)
            base_name = os.path.splitext(image_name)[0]
            file_path = os.path.join(self.output_folder, f"{base_name}.json")

            h, w = self.current_image.shape[:2]

            coco_data = {
                "info": {
                    "description": "SAM3 Annotator Export",
                    "date_created": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "licenses": [],
                "images": [{
                    "id": 1,
                    "file_name": image_name,
                    "width": w,
                    "height": h
                }],
                "categories": [],
                "annotations": []
            }

            for label, cat_id in self.categories.items():
                coco_data["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": ""
                })

            ann_id = 1
            for ann in self.canvas.annotations:
                if ann.label in self.categories:
                    cat_id = self.categories[ann.label]
                    coco_ann = ann.to_coco_annotation(ann_id, 1, cat_id)
                    if coco_ann:
                        coco_data["annotations"].append(coco_ann)
                        ann_id += 1

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"Saved: {base_name}.json ({len(self.canvas.annotations)} annotations)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")

    def _load_model(self):
        self.statusBar().showMessage("Loading SAM3 model with interactive predictor... Please wait.")
        QApplication.processEvents()

        try:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

            # Enable inst_interactivity for point prompt support
            self.model = build_sam3_image_model(
                bpe_path=bpe_path,
                enable_inst_interactivity=True  # Required for point prompts
            )
            confidence = self.confidence_slider.value() / 100.0
            self.processor = Sam3Processor(self.model, confidence_threshold=confidence)

            # Check if interactive predictor is available
            if self.model.inst_interactive_predictor is not None:
                self.model_status.setText("Model: Loaded (with Point Prompt)")
                self.model_status.setStyleSheet("color: #88ff88;")
                self.statusBar().showMessage("Model loaded with point prompt support!")
            else:
                self.model_status.setText("Model: Loaded (Text only)")
                self.model_status.setStyleSheet("color: #ffff88;")
                self.statusBar().showMessage("Model loaded (point prompt not available)")

            # If an image is already loaded, process it with the new model
            if self.pil_image is not None:
                self.statusBar().showMessage("Processing current image with loaded model...")
                QApplication.processEvents()
                self.inference_state = self.processor.set_image(self.pil_image)
                self.statusBar().showMessage("Model loaded and image processed!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.statusBar().showMessage("Failed to load model.")

    def _load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if file_path:
            self.image_list = [file_path]
            self.current_image_index = 0
            self._open_image(file_path)

    def _load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG']
            images = []
            for ext in extensions:
                images.extend(Path(folder_path).glob(ext))

            if images:
                self.image_list = sorted([str(p) for p in images])
                self.current_image_index = 0

                # Auto-set output folder only if not already set
                if self.output_folder is None:
                    self.output_folder = folder_path
                    display_path = folder_path if len(folder_path) < 30 else "..." + folder_path[-27:]
                    self.output_folder_label.setText(display_path)
                    self.output_folder_label.setStyleSheet("color: #88ff88;")
                    self.output_folder_label.setToolTip(folder_path)

                self._open_image(self.image_list[0])
                self.statusBar().showMessage(f"Loaded {len(self.image_list)} images from folder")
            else:
                QMessageBox.warning(self, "Warning", "No images found in folder")

    def _prev_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self._open_image(self.image_list[self.current_image_index])

    def _next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self._open_image(self.image_list[self.current_image_index])

    def _open_image(self, file_path: str):
        try:
            self.pil_image = Image.open(file_path).convert("RGB")
            self.current_image = np.array(self.pil_image)
            self.current_image_path = file_path

            self.original_canvas.set_image(self.current_image)
            self.canvas.set_image(self.current_image)

            if self.image_list:
                self.image_info_label.setText(f"{self.current_image_index + 1}/{len(self.image_list)}")

            if self.processor is not None:
                self.statusBar().showMessage("Processing image...")
                QApplication.processEvents()
                self.inference_state = self.processor.set_image(self.pil_image)
                self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")
            else:
                self.statusBar().showMessage(f"Loaded (no model): {os.path.basename(file_path)}")

            self.setWindowTitle(f"SAM3 Annotator - {os.path.basename(file_path)}")
            self.ann_list.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def _update_confidence(self, value: int):
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        if self.processor is not None:
            self.processor.confidence_threshold = confidence

    def _text_prompt(self):
        if self.processor is None:
            QMessageBox.warning(self, "Warning", "Please load the model first")
            return

        if self.inference_state is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        prompt = self.text_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a text prompt")
            return

        self.current_text_prompt = prompt
        self.statusBar().showMessage(f"Generating masks for '{prompt}'...")
        QApplication.processEvents()

        try:
            self.processor.reset_all_prompts(self.inference_state)
            self.inference_state = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=prompt
            )

            if "masks" in self.inference_state:
                masks = self.inference_state["masks"]
                scores = self.inference_state["scores"]

                mask_list = []
                score_list = []
                for i in range(len(masks)):
                    mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                    score = scores[i].item()
                    mask_list.append(mask)
                    score_list.append(score)

                self.canvas.set_temp_masks(mask_list, score_list)
                self.mask_selector.setMaximum(max(0, len(mask_list) - 1))
                self.statusBar().showMessage(f"Found {len(mask_list)} masks for '{prompt}'")
            else:
                self.statusBar().showMessage("No masks found")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Text prompt failed:\n{str(e)}")

    def _toggle_box_mode(self, enabled: bool):
        self.canvas.box_mode = enabled
        if enabled:
            self.edit_mode_cb.setChecked(False)
            self.point_mode_cb.setChecked(False)
            self.statusBar().showMessage("Box mode: Drag to draw a box on the result image")

    def _toggle_point_mode(self, enabled: bool):
        self.canvas.point_mode = enabled
        if enabled:
            self.edit_mode_cb.setChecked(False)
            self.box_mode_cb.setChecked(False)
            self.statusBar().showMessage("Point mode: Left click=positive (green), Right click=negative (red)")

    def _toggle_edit_mode(self, enabled: bool):
        self.canvas.edit_mode = enabled
        if enabled:
            self.box_mode_cb.setChecked(False)
            self.point_mode_cb.setChecked(False)
            self.statusBar().showMessage("Edit mode: Click on a mask to edit/delete")

    def _on_box_drawn(self, x1: int, y1: int, x2: int, y2: int):
        """Handle box drawn on canvas."""
        self.canvas.current_box = (x1, y1, x2, y2)
        self.canvas._update_display()
        self.statusBar().showMessage(f"Box drawn: ({x1}, {y1}) to ({x2}, {y2}). Click 'Generate from Box'")

    def _box_prompt(self):
        """Generate masks using box prompt - only within box region."""
        if self.processor is None:
            QMessageBox.warning(self, "Warning", "Please load the model first")
            return

        if self.inference_state is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        if self.canvas.current_box is None:
            QMessageBox.warning(self, "Warning", "Please draw a box first (enable Box Mode)")
            return

        x1, y1, x2, y2 = self.canvas.current_box
        h, w = self.current_image.shape[:2]

        # Use text prompt if available, otherwise use generic prompt
        prompt = self.text_input.text().strip() or "object"
        self.current_text_prompt = prompt

        self.statusBar().showMessage(f"Generating masks for '{prompt}' within box...")
        QApplication.processEvents()

        try:
            # Generate masks using text prompt
            self.processor.reset_all_prompts(self.inference_state)
            self.inference_state = self.processor.set_text_prompt(
                state=self.inference_state,
                prompt=prompt
            )

            if "masks" in self.inference_state:
                masks = self.inference_state["masks"]
                scores = self.inference_state["scores"]

                mask_list = []
                score_list = []

                for i in range(len(masks)):
                    mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                    score = scores[i].item()

                    # Filter: only keep masks that overlap significantly with box
                    # Create box mask
                    box_mask = np.zeros_like(mask)
                    box_mask[y1:y2, x1:x2] = 1

                    # Calculate overlap
                    intersection = np.logical_and(mask, box_mask).sum()
                    mask_area = mask.sum()

                    if mask_area > 0:
                        # Keep mask if >50% of it is inside the box
                        overlap_ratio = intersection / mask_area
                        if overlap_ratio > 0.5:
                            # Clip mask to box region
                            clipped_mask = np.logical_and(mask, box_mask).astype(np.uint8)
                            if clipped_mask.sum() > 100:  # Minimum area threshold
                                mask_list.append(clipped_mask)
                                score_list.append(score)

                if mask_list:
                    self.canvas.set_temp_masks(mask_list, score_list)
                    self.mask_selector.setMaximum(max(0, len(mask_list) - 1))
                    self.statusBar().showMessage(f"Found {len(mask_list)} masks within box")
                else:
                    self.statusBar().showMessage("No masks found within box region")
            else:
                self.statusBar().showMessage("No masks found")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Box prompt failed:\n{str(e)}")

    def _clear_box(self):
        """Clear the current box."""
        self.canvas.current_box = None
        self.canvas._update_display()
        self.statusBar().showMessage("Box cleared")

    def _on_point_added(self):
        """Handle point added to canvas - update status label."""
        n_pos = len(self.canvas.positive_points)
        n_neg = len(self.canvas.negative_points)
        self.point_status_label.setText(f"Points: {n_pos} pos, {n_neg} neg")

    def _clear_points(self):
        """Clear all prompt points."""
        self.canvas.clear_points()
        self.point_status_label.setText("Points: 0 pos, 0 neg")
        self.statusBar().showMessage("Points cleared")

    def _point_prompt(self):
        """Generate masks using point prompts with SAM3 interactive predictor."""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load the model first")
            return

        if self.inference_state is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        if not self.canvas.has_points():
            QMessageBox.warning(self, "Warning", "Please add at least one point (enable Point Mode)")
            return

        # Check if interactive predictor is available
        if self.model.inst_interactive_predictor is None:
            QMessageBox.warning(self, "Warning", "Interactive predictor not available in this model")
            return

        # Check if sam2_backbone_out is available (required for point prompts)
        if "sam2_backbone_out" not in self.inference_state.get("backbone_out", {}):
            QMessageBox.warning(self, "Warning",
                "SAM2 backbone features not available.\n"
                "Make sure model was loaded with enable_inst_interactivity=True")
            return

        self.statusBar().showMessage("Generating masks from points...")
        QApplication.processEvents()

        try:
            # Prepare point coordinates and labels
            points = []
            labels = []

            for px, py in self.canvas.positive_points:
                points.append([px, py])
                labels.append(1)  # Positive (foreground)

            for px, py in self.canvas.negative_points:
                points.append([px, py])
                labels.append(0)  # Negative (background)

            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)

            # Use model.predict_inst which properly uses inference_state
            masks_np, scores_np, _ = self.model.predict_inst(
                inference_state=self.inference_state,
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True
            )

            # Store as proposals
            self.proposal_masks = []
            self.proposal_scores = []

            mask_list = []
            score_list = []

            for i in range(masks_np.shape[0]):
                mask = masks_np[i].astype(np.uint8)
                score = float(scores_np[i])
                mask_list.append(mask)
                score_list.append(score)
                # Store each mask as a separate proposal
                self.proposal_masks.append([mask])
                self.proposal_scores.append([score])

            if mask_list:
                # Select best proposal by default (highest score)
                best_idx = int(np.argmax(score_list))
                self.selected_proposal_idx = best_idx

                self.canvas.set_temp_masks([mask_list[best_idx]], [score_list[best_idx]])
                self.mask_selector.setMaximum(0)

                # Update proposal buttons
                self._update_proposal_buttons(mask_list, score_list)

                self.statusBar().showMessage(f"Found {len(mask_list)} proposals. Use 1-4 to select.")
            else:
                self.statusBar().showMessage("No masks found")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Point prompt failed:\n{str(e)}")

    def _update_proposal_buttons(self, masks: List[np.ndarray], scores: List[float]):
        """Update proposal button text with scores."""
        for i, btn in enumerate(self.proposal_btns):
            if i < len(scores):
                btn.setText(f"{i+1}: {scores[i]:.2f}")
                btn.setEnabled(True)
            else:
                btn.setText(f"{i+1}")
                btn.setEnabled(False)

    def _select_proposal(self, idx: int):
        """Select a proposal mask by index."""
        if idx >= len(self.proposal_masks):
            return

        self.selected_proposal_idx = idx
        masks = self.proposal_masks[idx]
        scores = self.proposal_scores[idx]

        self.canvas.set_temp_masks(masks, scores)
        self.mask_selector.setMaximum(max(0, len(masks) - 1))
        self.statusBar().showMessage(f"Selected proposal {idx + 1}")

    def _reject_all(self):
        """Reject/clear all temporary masks and points."""
        self.canvas.clear_temp_masks()
        self.canvas.clear_points()
        self.canvas.current_box = None
        self.proposal_masks = []
        self.proposal_scores = []
        self.point_status_label.setText("Points: 0 pos, 0 neg")

        # Reset proposal buttons
        for i, btn in enumerate(self.proposal_btns):
            btn.setText(f"{i+1}")
            btn.setEnabled(True)

        self.statusBar().showMessage("Rejected all - cleared temp masks and prompts")

    def _delete_selected_annotation(self):
        """Delete selected annotation from list."""
        selected_items = self.ann_list.selectedItems()
        if not selected_items:
            self.statusBar().showMessage("No annotation selected. Select from list first.")
            return

        for item in selected_items:
            idx = self.ann_list.row(item)
            if 0 <= idx < len(self.canvas.annotations):
                self.canvas.remove_annotation(idx)

        self._update_annotation_list()
        self.statusBar().showMessage("Deleted selected annotation(s)")

    def _on_mask_clicked(self, x: int, y: int):
        idx = self.canvas.get_mask_at_point(x, y)
        if idx >= 0:
            ann = self.canvas.annotations[idx]
            dialog = LabelDialog(ann.label, list(self.categories.keys()), self)

            if dialog.exec_() == QDialog.Accepted:
                if dialog.result_action == 'delete':
                    self.canvas.remove_annotation(idx)
                    self._update_annotation_list()
                    self.statusBar().showMessage("Annotation deleted")
                elif dialog.result_action == 'edit':
                    new_label = dialog.get_label()
                    if new_label:
                        if new_label not in self.categories:
                            self.categories[new_label] = self.next_category_id
                            self.next_category_id += 1
                        ann.label = new_label
                        self.canvas._update_display()
                        self._update_annotation_list()
                        self.statusBar().showMessage(f"Label changed to '{new_label}'")
        else:
            self.statusBar().showMessage("No mask at this location")

    def _accept_selected_mask(self):
        if not self.canvas.temp_masks:
            self.statusBar().showMessage("No masks to accept. Generate masks first.")
            return

        idx = self.mask_selector.value()
        if idx >= len(self.canvas.temp_masks):
            idx = 0  # Default to first mask

        label = self.current_text_prompt if self.current_text_prompt else "object"

        if label not in self.categories:
            self.categories[label] = self.next_category_id
            self.next_category_id += 1

        mask = self.canvas.temp_masks[idx]
        score = self.canvas.temp_scores[idx]
        annotation = Annotation(mask, label, score)
        self.canvas.add_annotation(annotation)

        # Clear temp masks and points after accepting
        self.canvas.clear_temp_masks()
        self.canvas.clear_points()
        self.point_status_label.setText("Points: 0 pos, 0 neg")
        self.proposal_masks = []
        self.proposal_scores = []

        # Reset proposal buttons
        for i, btn in enumerate(self.proposal_btns):
            btn.setText(f"{i+1}")
            btn.setEnabled(True)

        self._update_annotation_list()
        self.statusBar().showMessage(f"Added: {label} (score: {score:.2f}). Press A to accept more or continue annotating.")

    def _accept_all_masks(self):
        if not self.canvas.temp_masks:
            QMessageBox.warning(self, "Warning", "No temporary masks")
            return

        label = self.current_text_prompt if self.current_text_prompt else "object"

        if label not in self.categories:
            self.categories[label] = self.next_category_id
            self.next_category_id += 1

        count = len(self.canvas.temp_masks)
        for mask, score in zip(self.canvas.temp_masks, self.canvas.temp_scores):
            annotation = Annotation(mask, label, score)
            self.canvas.add_annotation(annotation)

        self.canvas.clear_temp_masks()
        self._update_annotation_list()
        self.statusBar().showMessage(f"Added {count} annotations with label '{label}'")

    def _clear_temp_masks(self):
        self.canvas.clear_temp_masks()
        self.statusBar().showMessage("Temp masks cleared")

    def _update_annotation_list(self):
        self.ann_list.clear()
        for i, ann in enumerate(self.canvas.annotations):
            item = QListWidgetItem(f"{i}: {ann.label} ({ann.score:.2f})")
            color = QColor(*ann.color)
            item.setBackground(QBrush(color))
            item.setForeground(QBrush(Qt.white if color.lightness() < 128 else Qt.black))
            self.ann_list.addItem(item)

    def _on_annotation_double_clicked(self, item: QListWidgetItem):
        idx = self.ann_list.row(item)
        if 0 <= idx < len(self.canvas.annotations):
            ann = self.canvas.annotations[idx]
            dialog = LabelDialog(ann.label, list(self.categories.keys()), self)

            if dialog.exec_() == QDialog.Accepted:
                if dialog.result_action == 'delete':
                    self.canvas.remove_annotation(idx)
                    self._update_annotation_list()
                elif dialog.result_action == 'edit':
                    new_label = dialog.get_label()
                    if new_label:
                        if new_label not in self.categories:
                            self.categories[new_label] = self.next_category_id
                            self.next_category_id += 1
                        ann.label = new_label
                        self.canvas._update_display()
                        self._update_annotation_list()

    def _clear_annotations(self):
        reply = QMessageBox.question(
            self, "Confirm", "Clear all annotations?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.canvas.clear_annotations()
            self.ann_list.clear()
            self.statusBar().showMessage("All annotations cleared")

    def _update_opacity(self, value: int):
        self.canvas.mask_opacity = value / 100.0
        self.canvas._update_display()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, Qt.white)
    palette.setColor(palette.ToolTipText, Qt.white)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, Qt.white)
    palette.setColor(palette.BrightText, Qt.red)
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, Qt.black)
    app.setPalette(palette)

    window = SAM3Annotator()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
