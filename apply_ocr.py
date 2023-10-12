

import json
import sys

from src.controllers.extract_pdf_page import extract_single_page, extract_all_pages
from src.controllers.extract_raw_columns import extract_columns_from_page
from src.models.extract_area_rect import ExtractAreaRect

with open("./assets/rectangles.json", "r") as f:
    jsonRectangles = json.load(f)

areasToExtract = [ExtractAreaRect.fromJson(rect) for rect in jsonRectangles]

pdf_file = "assets/lind_pdf.pdf"

extracted_text = []

debug = sys.argv[1] == "debug"

image = extract_single_page(pdf_file, 0, debug)
extracted_text = extract_columns_from_page(image, areasToExtract)

# Save the JSON data to a file or do further processing
with open("output.json", "w") as json_file:
    json.dump(extracted_text, json_file, indent=4)
