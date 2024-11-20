import argparse
from inference import infer

def main():
    parser = argparse.ArgumentParser(description='Vehicle Tracking')
    parser.add_argument('--video_path', type=str, default='sprites/v1.mp4', help='Path to input video file')
    parser.add_argument('--output_path', type=str, help='Path to save the output video')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (cpu/gpu)')
    parser.add_argument('--model_path', type=str, default='models/m1/weights/best.pt', help='Path to the trained model file')
    parser.add_argument('--calibration_points', type=str, default='data/points/calibration_points.yml', help='Path to the YAML file containing points')

    args = parser.parse_args()
    infer(args.video_path, args.output_path, args.device, args.model_path, args.calibration_points)

if __name__ == "__main__":
    main()
