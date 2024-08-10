import argparse
import sys



import realtime as run_realtime_detection
from scripts.static_detection import run_static_detection

def main():
    parser = argparse.ArgumentParser(description='Face Detection using AWS Rekognition')
    parser.add_argument('--mode', choices=['realtime', 'static'], required=True, help='Mode of operation: "realtime" for webcam detection, "static" for image detection.')
    parser.add_argument('--image_path', type=str, help='Path to the input image for static detection.')
    parser.add_argument('--output_path', type=str, help='Path to save the output image for static detection.')

    args = parser.parse_args()

    if args.mode == 'realtime':
        run_realtime_detection()
    elif args.mode == 'static':
        if not args.image_path or not args.output_path:
            print("Please provide both --image_path and --output_path for static detection.")
            return
        run_static_detection(args.image_path, args.output_path)

if __name__ == "__main__":
    main()
