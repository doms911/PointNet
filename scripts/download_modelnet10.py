import os
import urllib.request
import zipfile

def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\rDownloading: {percent}%", end='')
    
def extract_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print(f"\rExtracting: {percent}%", end='')

url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
target_dir = "../data/raw"
zip_path = os.path.join(target_dir, "ModelNet10.zip")

os.makedirs(target_dir, exist_ok=True)

if not os.path.exists(zip_path) and not os.path.exists(os.path.join(target_dir, "ModelNet10")):
    print(f"Downloading ModelNet10 dataset from {url}...")
    urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
    print("Download complete.")
else:
    print("ModelNet10 dataset already downloaded and extracted.")

extract_dir = os.path.join(target_dir, "ModelNet10")
if not os.path.exists(extract_dir):
    print("Extracting ModelNet10 dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete.")
else:
    print("ModelNet10 dataset already extracted.")
    
if os.path.exists(zip_path):
    print("Removing zip file...")
    os.remove(zip_path)
    print("Cleanup complete.")
    
macosx_dir = os.path.join(target_dir, "__MACOSX")

if os.path.exists(macosx_dir):
    print("Removing __MACOSX directory...")
    os.remove(macosx_dir) if os.path.isfile(macosx_dir) else os.system(f"rm -rf '{macosx_dir}'")
    print("__MACOSX removed.")

