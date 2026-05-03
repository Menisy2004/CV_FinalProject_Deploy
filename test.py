import urllib.request
import os

os.makedirs("sam_model", exist_ok=True)

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
output_path = "sam_model/sam_vit_b.pth"

urllib.request.urlretrieve(url, output_path)

print("Download complete!")