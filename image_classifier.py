import fitz  # PyMuPDF
from PIL import Image
import io
import os
 
# Path to your PDF and output directory for saving images.
pdf_path = "/content/PS_2.1_011_1756_01 (2).pdf"
output_dir = "saved_images"
 
# Create output directory if it doesn't exist.
os.makedirs(output_dir, exist_ok=True)
 
doc = fitz.open(pdf_path)
num_pages = doc.page_count
print(f"Total pages: {num_pages}")
 
scale = 7.0  # Increase scale for higher resolution
mat = fitz.Matrix(scale, scale)
 
for i in range(num_pages):
    page = doc.load_page(i)
    image_list = page.get_images(full=True)
    if image_list:  # Only extract the whole page if an image is found on this page.
        print(f"Page {i+1}: Found {len(image_list)} image(s), extracting full page...")
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("ppm")
        img = Image.open(io.BytesIO(img_bytes))
        # Save the rendered page image.
        output_path = os.path.join(output_dir, f"saved_image_page_{i+1}.png")
        img.save(output_path)
        print(f"Saved page {i+1} as: {output_path}")
    else:
        print(f"Page {i+1}: No images found, skipping extraction.")
