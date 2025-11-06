import os
import random
from PIL import Image

data_dir = "data/assets/imagen1"
new_data_dir = "new_images"
floor_dir = os.path.join(data_dir, "floor")

# Load all available floor images
floor_images = [Image.open(os.path.join(floor_dir, f)).convert("RGBA")
                for f in os.listdir(floor_dir)
                if f.endswith(".png")]

if not floor_images:
    raise RuntimeError("No floor images found in data/floor/")

# Iterate through all class folders except floor
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path) or class_name == "floor":
        continue

    if class_name in {"dragon", "opened", "sleeping", "wolf", "wall", "lava"}:
        continue

    # Prepare save directory
    save_dir = os.path.join(new_data_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(class_path):
        if not filename.endswith(".png"):
            continue

        entity_path = os.path.join(class_path, filename)
        entity_img = Image.open(entity_path).convert("RGBA")

        
        floor_img = random.choice(floor_images)

        # Ensure both are same size — resize if needed
        if floor_img.size != entity_img.size:
            floor_img = floor_img.resize(entity_img.size)

        # Composite entity on top of floor
        combined = Image.alpha_composite(floor_img, entity_img)

        # Save in new_data/<class_name>/
        save_path = os.path.join(save_dir, filename)
        combined.save(save_path)

        print(f"Saved: {save_path}")
