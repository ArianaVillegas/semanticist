import argparse
from pathlib import Path
from analysis_lib import FeatureSpaceAnalyzer

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run SlotFormer Feature Space Analysis.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./model-199.ckpt',
        help='Path to the trained SlotFormer model checkpoint.'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=20,
        help='Number of images to analyze from the dataset.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory to save all analysis results.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run the analysis on (e.g., "cuda" or "cpu").'
    )
    parser.add_argument(
        '--no_synthetic',
        action='store_true',
        help='If set, do not include synthetic images in the analysis.'
    )

    args = parser.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    analyzer = FeatureSpaceAnalyzer(
        model_path=args.model_path,
        results_dir=args.results_dir,
        device=args.device
    )
    
    analyzer.run_full_analysis(
        num_images=args.num_images,
        use_synthetic=not args.no_synthetic
    )

if __name__ == "__main__":
    main()
