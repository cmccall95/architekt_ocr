import os
import re
from typing import List, Dict
from PIL import Image, ImageDraw
import pytesseract

from src.models.column_name import ColumnName
from src.models.extract_area_rect import ExtractAreaRect

def _clean_text(column: ColumnName, text: str):
    if(column == ColumnName.id):
        pattern = r'\s*o\s*$'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif(column == ColumnName.drawing):
        pattern = r'\b(process|medium|sequence|unit|code|number)\b|\s+'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.sheet):
        pattern = r'\b(sheet)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.area):
        pattern = r'\b(area)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.pipe_spec):
        pattern = r'\b(pipe|-|spec|speg)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.p_and_id):
        pattern =  r'^P&ID-No:'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.item):
        pattern = r'\b(item)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.tag):
        pattern = r'\b(tag)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.quantity):
        pattern = r'\b(quantity)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.nps):
        pattern = r'\b(nps)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.material_description):
        pattern = r'\b(material|description)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
    pattern = r'[|\]]+'
    return re.sub(pattern, '', text).strip()

def _store_image_with_rectangles(image: any):
    directory = "./debug"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    image.save(f"{directory}/image_with_rectangles.png")
    
def _store_cropped_rectangles(image: any, column: str):
    directory = "./debug/cropped_rectangles/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    image.save(f"{directory}/{column}.png")

# img: c2 image
def _extract_text_from_area(img: any, rectangles: List[ExtractAreaRect], debug: bool = False) -> Dict[str, List[str]]:
    extracted_text = {}
    
    img_for_draw = img.copy()
    draw = ImageDraw.Draw(img_for_draw)
    
    for i, rectangle in enumerate(rectangles):
        column_name = rectangle.column_name
        relativeX1 = rectangle.relative_x1
        relativeX2 = rectangle.relative_x2
        relativeY1 = rectangle.relative_y1
        relativeY2 = rectangle.relative_y2
        
        page_width = img.width
        page_height = img.height

        left_abs = relativeX1 * page_width
        top_abs = relativeY1 * page_height
        right_abs = relativeX2 * page_width
        bottom_abs = relativeY2 * page_height

        rect = (left_abs, top_abs, right_abs, bottom_abs)
        cropped_image = img.crop(rect)
        
        if(debug):
            draw.rectangle(rect, outline="blue", width=3)
            _store_cropped_rectangles(cropped_image, column_name.value)
        
        config = '--psm 6'
        if (column_name == ColumnName.material_description):
            config = '--psm 4' 
            
        text = pytesseract.image_to_string(cropped_image, config=config)
        
        cleaned_text = _clean_text(column_name, text)
        extracted_text[column_name.value] = cleaned_text

    # Save the image with rectangles for debug (optional)
    if(debug):
        _store_image_with_rectangles(img_for_draw)

    return extracted_text

# image: c2 image
def extract_columns_from_page(image: any,  rectangles: List[ExtractAreaRect]): 
    img_rgb = Image.fromarray(image)
    # img_rgb = Image.open('./debug/extracted_pages/0.png')
    return _extract_text_from_area(img_rgb, rectangles)