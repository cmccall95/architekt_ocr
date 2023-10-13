import os
import cv2
import numpy as np
import fitz

def _pixmap_to_numpy(pixmap):
    im = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)
    return np.ascontiguousarray(im[..., [2, 1, 0]])

def _remove_lines(image):
    override_line_thickness = 5
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.threshold(blackAndWhiteImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), override_line_thickness)
        
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), override_line_thickness)
    
    return image

def _write_debug_image(image, index):
    directory = "./debug/extracted_pages"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    cv2.imwrite(f"{directory}/{index}.png", image)
    
def _extract_page(page: any, debug: bool = False):
    imagePixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    image = _pixmap_to_numpy(imagePixmap)
    
    image = _remove_lines(image)
    
    if debug:
        _write_debug_image(image, page.number)
    
    return image

def extract_single_page(path: str, index: int, debug: bool = False):
    document = fitz.open(path)
    page = document.load_page(index)
    
    image = _extract_page(page, debug)
    
    document.close()
    
    return image

def extract_all_pages(path: str, debug: bool = False):
    document = fitz.open(path)
    pages = document.page_count
    
    images = []
    for i in range(pages):
        page = document.load_page(i)
        image = _extract_page(page, debug)
        images.append(image)
    
    document.close()
    return images

