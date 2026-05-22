import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / ".grok" / "skills"))

from vision_analyzer_v6.scripts.vision_analyzer_v6 import VisionAnalyzerV6

def analyze_image(image_path: str, real_height_cm: float = None):
    analyzer = VisionAnalyzerV6(real_height_cm=real_height_cm)
    return analyzer.get_full_visual_analysis(image_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--height", type=float)
    args = parser.parse_args()
    result = analyze_image(args.image, real_height_cm=args.height)
    print(result.status)