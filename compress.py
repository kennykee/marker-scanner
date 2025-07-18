import os
from PIL import Image
# Directory containing the images
cwd = os.getcwd()
max_size = 1280  # Maximum size for the longest side of the image

for filename in os.listdir(cwd):
	if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
		path = os.path.join(cwd, filename)
		try:
			with Image.open(path) as img:
				width, height = img.size
				if max(width, height) <= max_size:
					print(f"Skipping {filename}, size is already {width}x{height}")
					continue

				if width > height:
					new_width = max_size
					new_height = int(max_size * height / width)
				else:
					new_height = max_size
					new_width = int(max_size * width / height)

				resized_img = img.resize((new_width, new_height), Image.LANCZOS)
				resized_img.save(path, optimize=True, quality=85)
				print(f"Resized {filename} to {new_width}x{new_height}")
		except Exception as e:
			print(f"Error processing {filename}: {e}")