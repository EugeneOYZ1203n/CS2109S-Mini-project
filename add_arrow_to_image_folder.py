from PIL import Image
import os

# === CONFIG ===
entity_dir = "new_images/metalbox"       # folder with N base images
arrow_dir = "arrow_only_images"          # folder with 4 arrow images (transparent)
output_dir = "arrow_composite_images" # where to save the combined results
entity_name = "metalbox"

os.makedirs(output_dir, exist_ok=True)

# Load all arrows once
arrow_images = [(os.path.splitext(f)[0], Image.open(os.path.join(arrow_dir, f)).convert("RGBA"))
                for f in os.listdir(arrow_dir)
                if f.lower().endswith((".png", ".jpg"))]

for i, (arrow_name, arrow_image) in enumerate(arrow_images):
    output_dir_arrow = os.path.join(output_dir, f"{entity_name}_{arrow_name.split("_",1)[1]}")
    for entity_file in os.listdir(entity_dir):
        if not entity_file.lower().endswith((".png", ".jpg")):
            continue

        entity_path = os.path.join(entity_dir, entity_file)
        entity_img = Image.open(entity_path).convert("RGBA")

        arrow_resized = arrow_image.resize(entity_img.size, Image.Resampling.LANCZOS)

        # Overlay arrow on entity using alpha compositing
        combined = Image.alpha_composite(entity_img, arrow_resized)

        # Save result
        base_name, ext = os.path.splitext(entity_file)
        out_path = os.path.join(output_dir_arrow, f"{base_name}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.save(out_path)

print("✅ Done: All arrow-entity composites created!")