import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TrafficSignDetector:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)
        self.class_names = ['traffic_light', 'stop_sign', 'speed_limit', 'crosswalk']
        self.colors = {
            0: (255, 0, 0),    # Red for traffic light
            1: (0, 0, 255),    # Blue for stop sign
            2: (0, 255, 0),    # Green for speed limit
            3: (255, 255, 0)   # Yellow for crosswalk
        }
        
    def detect(self, image_path, conf_threshold=0.5):
        """Run detection on image"""
        results = self.model(image_path, conf=conf_threshold)
        return results[0]
    
    def visualize_results(self, results, image_path, save_path=None):
        """Create professional visualization of detection results"""
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title("Original Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Detection results
        ax2.imshow(img_rgb)
        ax2.set_title("Detection Results", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        detections = []
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls_id = int(box.cls)
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor=np.array(self.colors[cls_id])/255,
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                # Add label
                label = f"{self.class_names[cls_id]} {conf:.2f}"
                ax2.text(x1, y1-5, label, 
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor=np.array(self.colors[cls_id])/255,
                                 alpha=0.8),
                        color='white', fontsize=10, fontweight='bold')
                
                detections.append({
                    'class': self.class_names[cls_id],
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        # Add detection summary
        summary_text = f"Detected {len(detections)} object(s)"
        if detections:
            summary_text += ":\n"
            class_counts = {}
            for det in detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
            for cls, count in class_counts.items():
                summary_text += f"• {count} {cls}(s)\n"
        
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            save_dir = Path("results/visualizations")
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / save_path, dpi=100, bbox_inches='tight')
            print(f"Visualization saved to: {save_dir / save_path}")
        
        plt.show()
        return detections
    
    def process_batch(self, image_folder, output_folder="results/batch"):
        """Process multiple images"""
        image_folder = Path(image_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        results_summary = []
        for img_path in image_folder.glob("*.jpg"):
            print(f"Processing: {img_path.name}")
            results = self.detect(img_path)
            detections = self.visualize_results(
                results, img_path, 
                save_path=f"batch_{img_path.stem}.png"
            )
            results_summary.append({
                'image': img_path.name,
                'detections': detections
            })
        
        return results_summary
