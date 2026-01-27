import os
from PIL import Image
import random
import matplotlib.pyplot as plt

# Set base path for data - works from any working directory
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'Data')

batches = os.listdir(os.path.join(data_path, "Raw"))
# background_use_num = 1
# head_use_num = 1
size = 128
x = 0
y = 0
im_num = 0


# Use https://online.photoscissors.com/ to cut out head

def generating_waldos(use_background=True, variations_per_combo=2):
    size = 128
    im_num = 0
    
    # Get image files (not subdirectories)
    heads_path = os.path.join(data_path, "Clean/OnlyWaldoHeads")
    bg_path = os.path.join(data_path, "Clean/ClearedWaldos")
    # Print the data_path for debugging
    print(f"Data path: {data_path}")
    
    # Get all image files in heads directory
    head_files = [f for f in os.listdir(heads_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Get all image files in background directory
    bg_files = [f for f in os.listdir(bg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(head_files)} head images and {len(bg_files)} background images")
    print(f"Will generate {len(head_files) * len(bg_files) * variations_per_combo} total images")
    
    for head_name in head_files:
        for back_name in bg_files:
            for _ in range(variations_per_combo):
            # Rotate image
                head_file = os.path.join(heads_path, head_name)
                if random.randint(0, 9) < 5:
                    num = random.randint(-15, 15)
                    foreground = Image.open(head_file).rotate(num)
                else:
                    foreground = Image.open(head_file)

                if random.randint(0, 9) < 7:
                    scale = random.uniform(0.8, 1.5)
                    w, h = foreground.size
                    foreground = foreground.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                
                if use_background:
                    background = Image.open(os.path.join(bg_path, back_name))
                else:
                    background = Image.open(os.path.join(data_path, "Clean/black_background.jpg"))

                # Crop background and place foreground on top
                bck_w, bck_h = background.size
                frg_w, frg_h = foreground.size

                bck_x = random.randint(0, max(0, bck_w - size))
                bck_y = random.randint(0, max(0, bck_h - size))
                
                frg_x = random.randint(0, max(0, size - frg_w))
                frg_y = random.randint(0, max(0, size - frg_h))

                cropped = background.crop((bck_x, bck_y, min(bck_x + size, bck_w), min(bck_y + size, bck_h)))
                
                # Resize cropped image to ensure it's 64x64
                if cropped.size != (size, size):
                    cropped = cropped.resize((size, size), Image.LANCZOS)
                
                # Ensure output directories exist
                os.makedirs(os.path.join(data_path, "NotWaldo"), exist_ok=True)
                os.makedirs(os.path.join(data_path, "Waldo"), exist_ok=True)
                
                cropped.save(os.path.join(data_path, "NotWaldo", "n" + str(im_num) + ".jpg"))
                cropped.paste(foreground, (frg_x, frg_y), foreground)
                cropped.save(os.path.join(data_path, "Waldo", str(im_num)+str(use_background) + ".jpg"))
                im_num += 1
    
    print(f"Generated {im_num} training images")

