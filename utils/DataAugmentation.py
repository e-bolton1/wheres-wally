import os
import gc
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

def generating_waldos(use_background=True, variations_per_combo=2, max_images=None, 
                      max_heads=None, max_backgrounds=None, sample_ratio=1.0):
    """
    Generate Waldo training images with dynamic control over dataset size.
    
    Args:
        use_background: Whether to use real backgrounds or black background
        variations_per_combo: Number of variations per head/background combo
        max_images: Maximum total images to generate (overrides other limits)
        max_heads: Maximum number of head images to use (None = use all)
        max_backgrounds: Maximum number of background images to use (None = use all)
        sample_ratio: Ratio of available combos to use (0.0-1.0, default 1.0 = all)
    """
    size = 128
    im_num = 0
    
    # Get image files (not subdirectories)
    heads_path = os.path.join(data_path, "Clean/OnlyWaldoHeads")
    bg_path = os.path.join(data_path, "Clean/ClearedWaldos")
    # Print the data_path for debugging
    print(f"Data path: {data_path}")
    
    # Get all image files in heads directory
    all_head_files = [f for f in os.listdir(heads_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_bg_files = [f for f in os.listdir(bg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Apply limits to head and background files
    if max_heads and max_heads < len(all_head_files):
        head_files = random.sample(all_head_files, max_heads)
    else:
        head_files = all_head_files
    
    if max_backgrounds and max_backgrounds < len(all_bg_files):
        bg_files = random.sample(all_bg_files, max_backgrounds)
    else:
        bg_files = all_bg_files
    
    # Apply sample ratio by randomly selecting combinations
    if sample_ratio < 1.0:
        total_combos = len(head_files) * len(bg_files)
        num_combos = max(1, int(total_combos * sample_ratio))
        print(f"Sampling {num_combos} out of {total_combos} possible combinations ({sample_ratio*100:.1f}%)")
        
        # Create list of all combinations and sample from it
        all_combos = [(h, b) for h in head_files for b in bg_files]
        selected_combos = random.sample(all_combos, num_combos)
        combos_to_use = selected_combos
    else:
        combos_to_use = [(h, b) for h in head_files for b in bg_files]
    
    # Calculate expected total images
    expected_total = len(combos_to_use) * variations_per_combo
    
    # Apply max_images limit if specified
    if max_images:
        expected_total = min(expected_total, max_images)
    
    print(f"Found {len(all_head_files)} head images, {len(all_bg_files)} background images")
    print(f"Using {len(head_files)} heads × {len(bg_files)} backgrounds × {variations_per_combo} variations")
    print(f"Will generate up to {expected_total} total images")
    
    # Ensure output directories exist before loop
    os.makedirs(os.path.join(data_path, "NotWaldo"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "Waldo"), exist_ok=True)
    
    for head_name, back_name in combos_to_use:
        for variation in range(variations_per_combo):
            # Check if we've reached max_images limit
            if max_images and im_num >= max_images:
                print(f"\nReached max_images limit of {max_images}")
                break
                
            # Use context managers to ensure images are properly closed
            head_file = os.path.join(heads_path, head_name)
                
                # Load and process foreground
            with Image.open(head_file) as head_img:
                if random.randint(0, 9) < 5:
                    num = random.randint(-15, 15)
                    foreground = head_img.rotate(num)
                else:
                    foreground = head_img.copy()

                if random.randint(0, 9) < 7:
                    scale = random.uniform(0.8, 1.5)
                    w, h = foreground.size
                    foreground = foreground.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                
                # Load background
                bg_file = os.path.join(bg_path, back_name) if use_background else os.path.join(data_path, "Clean/black_background.jpg")
                with Image.open(bg_file) as bg_img:
                    background = bg_img.copy()

                # Crop background and place foreground on top
                bck_w, bck_h = background.size
                frg_w, frg_h = foreground.size

                bck_x = random.randint(0, max(0, bck_w - size))
                bck_y = random.randint(0, max(0, bck_h - size))
                
                frg_x = random.randint(0, max(0, size - frg_w))
                frg_y = random.randint(0, max(0, size - frg_h))

                cropped = background.crop((bck_x, bck_y, min(bck_x + size, bck_w), min(bck_y + size, bck_h)))
                
                # Resize cropped image to ensure it's correct size
                if cropped.size != (size, size):
                    cropped = cropped.resize((size, size), Image.LANCZOS)
                
                # Convert RGBA to RGB before saving as JPEG
                if cropped.mode == 'RGBA':
                    cropped = cropped.convert('RGB')
                
                # Save NotWaldo version
                cropped.save(os.path.join(data_path, "NotWaldo", "n" + str(im_num) + ".jpg"))
                
                # Create Waldo version
                cropped.paste(foreground, (frg_x, frg_y), foreground)
                
                # Convert RGBA to RGB before saving as JPEG
                if cropped.mode == 'RGBA':
                    cropped = cropped.convert('RGB')
                    
                cropped.save(os.path.join(data_path, "Waldo", str(im_num)+str(use_background) + ".jpg"))
                
                # Clean up images from memory
                foreground.close()
                background.close()
                cropped.close()
                
                im_num += 1
                
                # Periodic garbage collection every 50 images
                if im_num % 50 == 0:
                    gc.collect()
                    print(f"Progress: {im_num}/{expected_total} images generated ({im_num*100//max(1, expected_total)}%)")
        
        # Early exit if max_images reached
        if max_images and im_num >= max_images:
            break
    
    print(f"\nGenerated {im_num} training images")
    print(f"  - {im_num} Waldo images saved to {os.path.join(data_path, 'Waldo')}")
    print(f"  - {im_num} NotWaldo images saved to {os.path.join(data_path, 'NotWaldo')}")

