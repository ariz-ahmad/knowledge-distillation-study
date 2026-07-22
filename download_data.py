import urllib.request
import tarfile
import os

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
filename = "imagenette2-320.tgz"

def progress(block, block_size, total):
    mb_done = block * block_size / 1024 / 1024
    mb_total = total / 1024 / 1024
    print(f"\r  {mb_done:.0f} MB / {mb_total:.0f} MB ({mb_done/mb_total*100:.1f}%)", end="")

if __name__ == "__main__":
    print("Downloading ImageNette (~1.4GB)...")
    urllib.request.urlretrieve(url, filename, reporthook=progress)

    print("\nExtracting...")
    with tarfile.open(filename) as tar:
        tar.extractall()

    os.remove(filename)
    print("✅ Done! Dataset saved to imagenette2-320/")
