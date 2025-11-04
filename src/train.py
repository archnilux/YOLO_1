import yaml
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import shutil

class TrafficSignTrainer:
    def __init__(self, data_path="data/yolo_dataset"):
        self.data_path = Path(data_path)
        self.model = None
        self.results_path = Path("results")
        self.results_path.mkdir(exist_ok=True)
        
        self.class_names = ['traffic_light', 'stop_sign', 'speed_limit', 'crosswalk']
        
    def prepare_dataset(self):
        """Prepare YOLO format dataset configuration"""
        data_config = {
            'path': str(self.data_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_path = self.data_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(data_config, f)
        
        print(f"Dataset config created at: {config_path}")
        return config_path
    
    def train(self, epochs=25, batch_size=16):
        """Train YOLOv8 model"""
        print("Initializing YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')  # Using nano version for speed
        
        print(f"Training for {epochs} epochs...")
        results = self.model.train(
            data=str(self.data_path / 'data.yaml'),
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            project='runs',
            name=f'traffic_signs_{datetime.now().strftime("%Y%m%d_%H%M")}',
            patience=5,
            save=True,
            plots=True
        )
        
        # Save best model to models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        best_model = Path("runs") / f'traffic_signs_{datetime.now().strftime("%Y%m%d_%H%M")}' / "weights" / "best.pt"
        if best_model.exists():
            shutil.copy(best_model, models_dir / "best.pt")
            print(f"Model saved to: {models_dir / 'best.pt'}")
        
        return results
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.model:
            self.model = YOLO("models/best.pt")
        
        print("Evaluating model performance...")
        results = self.model.val()
        
        # Save metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv8n",
            "dataset": "Traffic Signs",
            "performance": {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr)
            }
        }
        
        metrics_path = self.results_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print("\nModel Performance:")
        for metric, value in metrics["performance"].items():
            print(f"{metric:12} : {value:.3f}")
        (
        return metrics
