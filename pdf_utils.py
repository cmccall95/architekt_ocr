import os
import sys

import cv2
import fitz
import mimetypes
import numpy as np


def extract_pdf(pdf_file: str, limit: int):
    if not os.path.exists(pdf_file):
        raise "Provided file %s does not exist" % pdf_file

    if not mimetypes.guess_type(pdf_file)[0] == "application/pdf":
        raise "Provided file %s is not a valid PDF" % pdf_file

    document = fitz.open(pdf_file)
    dir_name = os.path.basename(pdf_file).split('.')[0]

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    i = 0
    for page in document:
        if i == limit:
            break
        table = crop_required_portions(
            _pixmap_to_numpy(page.get_pixmap(matrix=fitz.Matrix(3, 3)))
        )

        if table is not None:
            cv2.imwrite(dir_name + "/" + str(i) + "_0.png", table[0])
            cv2.imwrite(dir_name + "/" + str(i) + "_1.png", table[1])

        print(str(i) + "  Done")
        i += 1


def crop_required_portions(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(g_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    v_contours = _find_contours(thresh, (1, 10))
    h_contours = _find_contours(thresh, (10, 2))

    v_lines = sorted(
        [_max(v_contours, 3), _max(v_contours, 3), _max(v_contours, 3)],
        key=lambda x: x[0][0],
    )

    h_lines = sorted(
        [_max(h_contours, 2), _max(h_contours, 2), _max(h_contours, 2)],
        key=lambda x: x[0][1],
    )

    for lines in v_lines + h_lines:
        cv2.drawContours(img, [lines[1]], 0, (255, 255, 255), 3)

    return \
        img[h_lines[1][0][1]:h_lines[2][0][1], v_lines[0][0][0]:v_lines[2][0][0]], \
        img[h_lines[0][0][1]:h_lines[1][0][1], v_lines[1][0][0]:v_lines[2][0][0]]


def _max(_lines, _index):
    max_line = _lines[0]

    for line in _lines[1:]:
        if line[0][_index] > max_line[0][_index]:
            max_line = line

    _lines.remove(max_line)
    return max_line


def _pixmap_to_numpy(pixmap):
    im = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)
    return np.ascontiguousarray(im[..., [2, 1, 0]])


def _find_contours(_thresh, _size):
    contours = cv2.findContours(
        cv2.morphologyEx(
            _thresh,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, _size),
            iterations=2
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = contours[0] if len(contours) == 2 else contours[1]

    lines = []
    for c in contours:
        lines.append((cv2.boundingRect(c), c))

    return lines


def main():
    if len(sys.argv) < 3:
        raise "No enough arguments, Provide filename and limit of pages"

    extract_pdf(sys.argv[1], int(sys.argv[2]))


if __name__ == "__main__":
    main()
