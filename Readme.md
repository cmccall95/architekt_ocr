### Usage

To use this project, you can run the `apply_ocr.py` script. This script takes a PDF file as input, by default it is taking `assets/lind_pdf.pdf`, extracts text from the specified areas on each page using OCR, and saves the extracted text to a JSON file.

To run the script, use the following command:

```bash
python apply_ocr.py debug
```

The `debug` argument is optional. If you include the `debug` argument, the script will save the images processed and used by the OCR to the `./debug` directory.
