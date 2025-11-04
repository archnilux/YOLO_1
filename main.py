import argparse
from pathlib import Path
from src.train import TrafficSignTrainer
from src.inference import TrafficSignDetector
from src.utils import setup_directories, download_dataset

def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection with YOLOv8')
    parser.add_argument('--mode', choices=['train', 'inference', 'prepare'], 
                       default='train', help='Operation mode')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--model-path', type=str, help='Path to model for inference')
    parser.add_argument('--image', type=str, help='Image path for inference')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if args.mode == 'prepare':
        print("Downloading and preparing dataset...")
        dataset_path = download_dataset()
        print(f"Dataset ready at: {dataset_path}")
        
    elif args.mode == 'train':
        print("Starting training...")
        trainer = TrafficSignTrainer()
        trainer.prepare_dataset()
        trainer.train(epochs=args.epochs, batch_size=args.batch_size)
        trainer.evaluate()
        
    elif args.mode == 'inference':
        if not args.model_path or not args.image:
            print("Please provide --model-path and --image for inference")
            return
            
        detector = TrafficSignDetector(args.model_path)
        results = detector.detect(args.image)
        detector.visualize_results(results, args.image)

if __name__ == "__main__":
    main()