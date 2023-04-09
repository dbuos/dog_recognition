from drecg.feature_extraction.utils import extract_features_with_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputDir", help="Output directory", required=True)
    parser.add_argument("-m", "--model", help="Model name for feature extraction [ViT_L_16, ViT_LAION]", required=True)
    args = parser.parse_args()

    extract_features_with_model(root_dir=args.outputDir, model=args.model)
