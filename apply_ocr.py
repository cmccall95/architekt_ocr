

import json
import sys
import time

from src.controllers.extract_pdf_page import extract_single_page, extract_all_pages
from src.controllers.extract_raw_columns import extract_columns_from_page
from src.models.extract_area_rect import ExtractAreaRect

if len(sys.argv) < 4:
    print("Expected args: [pdf_path:str, rectangles_path:str, output_dir: str]")
    exit(1)
    
pdf_path = sys.argv[1]
rectangles_path = sys.argv[2]
output_dir = sys.argv[3]
index = None
debug = False

if len(sys.argv) > 4:
    if sys.argv[4].isnumeric():
        index = int(sys.argv[4])
    else:
        debug = sys.argv[4] == "debug"
    
if len(sys.argv) > 5:
    debug = sys.argv[5] == "debug"

extracted_text = []

start_time = time.time()
if index is not None:
    image = extract_single_page(pdf_path, index, debug)
    extracted_text.append(extract_columns_from_page(image, rectangles_path, debug))
else:
    images = extract_all_pages(pdf_path, debug)
    for i, image in enumerate(images):
        extracted_text.append(extract_columns_from_page(image, rectangles_path, debug))
        
with open(f"{output_dir}/output.json", "w") as json_file:
    json.dump(extracted_text, json_file, indent=4)
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
