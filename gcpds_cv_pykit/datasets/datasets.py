import kagglehub
import shutil
import os
from pathlib import Path

def download_and_prepare_dataset(
    kaggle_dataset: str
) -> str:
    """
    Download a Kaggle dataset using kagglehub and copy it to a local folder under 'datasets/{dataset_name}'.
    If the dataset is already present, skip download and copy.
    After copying, try to delete the original download folder.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (e.g., 'user/dataset/versions/x').

    Returns:
        str: Path to the prepared dataset folder.
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
    kaggle_path = kagglehub.dataset_download(kaggle_dataset)
    print("Path to downloaded dataset files:", kaggle_path)

    kaggle_path = Path(kaggle_path)
    if not target_path.exists():
        shutil.copytree(kaggle_path, target_path)
        print(f"Dataset copied to: {target_path}")
    else:
        print(f"Target folder '{target_path}' already exists, skipping copy.")

    # Try to delete the original download folder
    try:
        shutil.rmtree(kaggle_path)
        print(f"Original download folder '{kaggle_path}' deleted.")
    except Exception as e:
        print(f"Could not delete original download folder '{kaggle_path}': {e}")

    return str(target_path)

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
    kaggle_dataset: str = "lucasiturriago/feet-mamitas/versions/3"
) -> str:
    """
    Download and prepare the Feet Mamitas dataset from KaggleHub.

    Args:
        kaggle_dataset (str): KaggleHub dataset identifier (default: 'lucasiturriago/feet-mamitas/versions/3').

    Returns:
        str: Path to the prepared dataset folder.
    """
    return download_and_prepare_dataset(kaggle_dataset)