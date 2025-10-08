import kagglehub
import shutil
import os
from pathlib import Path

def download_and_prepare_dataset(
    kaggle_dataset: str
) -> str:
    """
    Download a Kaggle dataset using kagglehub.
    
    - If the downloaded folder is read-only (e.g., mounted in Kaggle /input), 
      use it directly without copying.
    - If the folder is writable, copy it to 'datasets/{dataset_name}' 
      and remove the original to free space.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier 
                              (e.g., 'user/dataset/versions/x').

    Returns:
        str: Path to the usable dataset folder.
    """
    cwd = Path.cwd()
    datasets_dir = cwd / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    # Extract dataset name (e.g., 'oxfordiiitpet' from 'lucasiturriago/oxfordiiitpet/versions/2')
    dataset_name = kaggle_dataset.split('/')[1]
    target_path = datasets_dir / dataset_name

    if target_path.exists() and any(target_path.iterdir()):
        print(f"Dataset already exists at: {target_path}")
        return str(target_path)

    print("Downloading dataset from KaggleHub...")
    kaggle_path = Path(kagglehub.dataset_download(kaggle_dataset))
    print("Path to downloaded dataset files:", kaggle_path)

    # Check if the folder is writable
    test_file = next(kaggle_path.rglob("*"), None)
    is_writable = test_file is not None and os.access(test_file, os.W_OK)

    if is_writable:
        # Copy to working datasets folder
        if not target_path.exists():
            shutil.copytree(kaggle_path, target_path)
            print(f"Dataset copied to: {target_path}")
        else:
            print(f"Target folder '{target_path}' already exists, skipping copy.")

        # Remove original folder
        try:
            shutil.rmtree(kaggle_path)
            print(f"Original folder '{kaggle_path}' deleted.")
        except Exception as e:
            print(f"Could not delete '{kaggle_path}': {e}")

        return str(target_path)
    else:
        # Just use original read-only folder
        print(f"Original dataset folder '{kaggle_path}' is read-only. Using it directly.")
        return str(kaggle_path)

def OxfordIITPet(
    kaggle_dataset: str = "lucasiturriago/oxfordiiitpet/versions/3"
) -> str:
    """
    Download and prepare the OxfordIITPet dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/oxfordiiitpet/versions/3').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)

def SeedGermination(
    kaggle_dataset: str = "lucasiturriago/seeds/versions/1"
) -> str:
    """
    Download and prepare the Seed Germination dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/seeds/versions/1').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)

def BreastCancer(
    kaggle_dataset: str = "lucasiturriago/breast-cancer-ss/versions/1"
) -> str:
    """
    Download and prepare the Breast Cancer Semantic Segmentation dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/breast-cancer-ss/versions/1').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)

def FeetMamitas(
    kaggle_dataset: str = "lucasiturriago/feet-mamitas/versions/4"
) -> str:
    """
    Download and prepare the Feet Mamitas dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/feet-mamitas/versions/4').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)

def OxfordIITPet_Crowd(
    kaggle_dataset: str = "lucasiturriago/oxfordiiitpet-multi-annotators/versions/4"
) -> str:
    """
    Download and prepare the OxfordIITPet Multiples Annotators dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/oxfordiiitpet-multi-annotators/versions/4').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)

def BreastCancer_Crowd(
    kaggle_dataset: str = "lucasiturriago/breast-cancer-multi-annotators/versions/4"
) -> str:
    """
    Download and prepare the Breast Cancer Multiples Annotators dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/breast-cancer-multi-annotators/versions/4').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)