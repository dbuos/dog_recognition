from drecg.data.utils import convert_to_hdf5
from pathlib import Path

if __name__ == '__main__':
    test_files = ["feat_extracted/laion_last_hidden/test_features.pt"]
    val_files = ["feat_extracted/laion_last_hidden/validation_features.pt"]
    train_files = ["feat_extracted/laion_last_hidden/train_features.pt"]
    train_augmented_files = train_files + ["feat_extracted/laion_last_hidden/train_features_augmented_p0.pt",
                                           "feat_extracted/laion_last_hidden/train_features_augmented_p1.pt"]
    dst_folder = "feat_extracted/laion_last_hidden_hdf5"
    # create path if not exists
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    convert_to_hdf5(source_files=test_files, dest_path=f"{dst_folder}/test_features.hdf5")
    convert_to_hdf5(source_files=val_files, dest_path=f"{dst_folder}/validation_features.hdf5")
    convert_to_hdf5(source_files=train_files, dest_path=f"{dst_folder}/train_features.hdf5")
    convert_to_hdf5(source_files=train_augmented_files, dest_path=f"{dst_folder}/train_features_augmented.hdf5")
