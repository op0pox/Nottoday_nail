import os
import json
from PIL import Image, ImageOps

target_dir = r"C:\Users\USER\Downloads\흰색배경데이터"

for file_name in os.listdir(target_dir):
    if file_name.lower().endswith(".json"):
        file_path = os.path.join(target_dir, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        base_name = os.path.splitext(file_name)[0]
        
        image_filename = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            potential_path = os.path.join(target_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_filename = base_name + ext
                break
        
        if image_filename:
            data["imagePath"] = image_filename
            image_path = os.path.join(target_dir, image_filename)
            
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                data["imageWidth"] = img.width
                data["imageHeight"] = img.height
            
            data["imageData"] = None
        
        if "shapes" in data:
            for shape in data["shapes"]:
                shape["label"] = "nail"
                
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)