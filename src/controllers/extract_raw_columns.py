import json
import os
import re
from typing import List, Dict
from PIL import Image, ImageDraw
import pytesseract

from src.models.column_name import ColumnName
from src.models.extract_area_rect import ExtractAreaRect

def _load_rectangles(path: str):
    with open(path, "r") as f:
        jsonRectangles = json.load(f)
        
    return [ExtractAreaRect.fromJson(rect) for rect in jsonRectangles]

def _clean_text(column: ColumnName, text: str):
    if(column == ColumnName.id):
        pattern = r'\s*o\s*$'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif(column == ColumnName.drawing):
        pattern = r'\b(process|medium|sequence|unit|code|number|drawing|no:\n)\b|\s+'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.sheet):
        pattern = r'\b(sheet|REV:\n)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.area):
        pattern = r'\b(area)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.pipe_spec):
        pattern = r'\b(pipe|-|spec|speg)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.p_and_id):
        pattern =  r'^P&ID-No:|reference|P&ID|\u2014|_'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.item):
        pattern = r'\b(item|fa|fr|er)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.tag):
        pattern = r'\b(tag)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.quantity):
        pattern = r'\b(quantity|qty)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.nps):
        pattern = r'\b(nps|^N\.S\. \(IN\)\n)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.material_description):
        pattern = r'\b(material|description|list|brication\n|\n\nrection)\b|\\'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.heat_trace):
        pattern = r'\b(heat|trace)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.insulation_type):
        pattern = r'\b(insulation|type)\b|='
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.insulation_thickness):
        pattern = r'\b(insulation|thickness)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    elif (column == ColumnName.process_line_list):
        pattern = r'\b(process|line|list)\b'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
        
    pattern = r'[|\]]+'
    return re.sub(pattern, '', text).strip()

def _process_bom_table(extracted_data: dict): 
    extracted_mto = {}
        
    if (ColumnName.item.value  in extracted_data):
        item_rows = extracted_data[ColumnName.item.value].split('\n')
        item_rows = [item for item in item_rows if item.strip()]
        del extracted_data[ColumnName.item.value]
        
        extracted_mto[ColumnName.item.value] = item_rows
        
    if (ColumnName.pos.value in extracted_data):
        pos_rows = extracted_data[ColumnName.pos.value].split('\n')
        pos_rows = [pos for pos in pos_rows if pos.strip()]
        del extracted_data[ColumnName.pos.value]
        
        extracted_mto[ColumnName.pos.value] = pos_rows
        
    if (ColumnName.ident.value in extracted_data):
        ident_rows = extracted_data[ColumnName.ident.value].split('\n')
        ident_rows = [ident for ident in ident_rows if ident.strip()]
        del extracted_data[ColumnName.ident.value]
        
        extracted_mto[ColumnName.ident.value] = ident_rows
        
    if (ColumnName.npd.value in extracted_data):
        npd_rows = extracted_data[ColumnName.npd.value].split('\n')
        npd_rows = [npd for npd in npd_rows if npd.strip()]
        del extracted_data[ColumnName.npd.value]
        
        extracted_mto[ColumnName.npd.value] = npd_rows
        
    if (ColumnName.quantity.value in extracted_data):
        quantity_rows = extracted_data[ColumnName.quantity.value].split('\n')
        quantity_rows = [quantity for quantity in quantity_rows if quantity.strip()]
        del extracted_data[ColumnName.quantity.value]
        
        extracted_mto[ColumnName.quantity.value] = quantity_rows
    
    if (ColumnName.nps.value in extracted_data):
        nps_rows = extracted_data[ColumnName.nps.value].split('\n')
        nps_rows = [nps for nps in nps_rows if nps.strip()]
        del extracted_data[ColumnName.nps.value]
        
        extracted_mto[ColumnName.nps.value] = nps_rows
    
    if (ColumnName.material_description.value in extracted_data):
        pattern = r'(S-STD\n|mm\)\n|in\. Length)'
        description_rows = re.split(pattern, extracted_data[ColumnName.material_description.value])
        del extracted_data[ColumnName.material_description.value]
        
        # Combine the removed parts from the split (S-STD\n, mm)\n, in. Length) back into the description.
        description_rows = [description_rows[i] + (description_rows[i + 1] if i + 1 < len(description_rows) else '') for i in range(0, len(description_rows), 2)]
        description_rows = [description.replace('\n', '') for description in description_rows] 
        
        extracted_mto[ColumnName.material_description.value] = description_rows
        
    data = []
    for i in range(len(list(extracted_mto.values())[0])):
        row = {**extracted_data}
        for key in extracted_mto:
            row[key] = extracted_mto[key][i] if i < len(extracted_mto[key]) else "-"
       
        data.append(row)
        
    return data

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
def _extract_text_from_area(img: any, rectangles: List[ExtractAreaRect], debug: bool = False):
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

    processed_data = _process_bom_table(extracted_text)
    return processed_data

# image: c2 image
def extract_columns_from_page(image: any,  rectangles_path: str, debug: bool = False): 
    rectangles = _load_rectangles(rectangles_path)
    img_rgb = Image.fromarray(image)
    
    return _extract_text_from_area(img_rgb, rectangles, debug)