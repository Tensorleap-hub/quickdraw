import os
import urllib.request
import pathlib

# The 20 classes that give you ~5-6 Million samples
classes = [
    "cat", "dog", "apple", "airplane", "face", "fish", "bread", "car", "cloud", "tree",
    "book", "eye", "flower", "house", "jail", "key", "line", "moon", "mountain", "star",
    "fork", "hospital", "lollipop", "sun", "suitcase", "tornado"
]

def download_data():
    # Create separate folders to keep things clean
    base_path = pathlib.Path("~/tensorleap/data/quickdraw_data").expanduser()
    os.makedirs(os.path.join(base_path, "npy"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "ndjson"), exist_ok=True)

    base_npy = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    base_ndjson = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

    print(f"Starting download for {len(classes)} classes...")

    for cls in classes:
        # URL encode the class name (handles spaces if you add 'smiley face' later)
        safe_cls = cls.replace(" ", "%20")

        # 1. Download .npy (Images)
        npy_url = f"{base_npy}{safe_cls}.npy"
        npy_path = f"{os.path.join(base_path, 'npy')}/{cls}.npy"
        if not os.path.exists(npy_path):
            print(f"Downloading Images (.npy): {cls}...")
            urllib.request.urlretrieve(npy_url, npy_path)

        # 2. Download .ndjson (Metadata)
        ndjson_url = f"{base_ndjson}{safe_cls}.ndjson"
        ndjson_path = f"{os.path.join(base_path, 'ndjson')}/{cls}.ndjson"
        if not os.path.exists(ndjson_path):
            print(f"Downloading Metadata (.ndjson): {cls}...")
            urllib.request.urlretrieve(ndjson_url, ndjson_path)
if __name__ == "__main__":
    download_data()