import os 
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

def resize_image(image_paths, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(outpath)
input_folder = "/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256"
output_folder = "/home/achraf/Desktop/workspace/SkinCanderDetection/Dataset/train512"
# Create a list of all images in training folder
images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,
        output_folder,
        (512, 512)
    ) for i in tqdm(images)
)