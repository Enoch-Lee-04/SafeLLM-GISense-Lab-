# diag_comparisons.py
import sys, itertools
from pathlib import Path
# Ensure project root is on sys.path
sys.path.append('.')
from scripts.evaluation.vlm_baseline_evaluation_fixed import FixedBaselineSafetyEvaluator

def main():
    evaluator = FixedBaselineSafetyEvaluator('Qwen/Qwen2-VL-2B-Instruct', device='cpu')
    image_folder = Path('data/samples/Sample SVI')
    images = sorted([str(p) for p in image_folder.glob('*.jpg')])[:4]
    pairs = list(itertools.combinations(images, 2))[:3]
    prompt = 'Which is safer? 1 or 2'
    print('Dumping first 3 raw comparison responses:')
    for i, (a, b) in enumerate(pairs, 1):
        resp = evaluator.compare_two_images(a, b, prompt)
        print(f"{i}: {Path(a).stem} vs {Path(b).stem} -> {resp!r}")

if __name__ == '__main__':
    main()
