For PDF image extraction use following command

```bash
python pdf_utils.py <input-pdf> <pages-count>
```

Use lesser pages-count for testing purpose

--------------

For applying OCR on the extracted images use following command
```bash
python main.py <input-image>
```

NOTE: only the images extracted by pdf_utils having name *_1.png can be used as input to main.py

### Usage New implementation

To use this project, you can run the `apply_ocr.py` script. This script takes a PDF file as input, by default it is taking `assets/lind_pdf.pdf`, extracts text from the specified areas on each page using OCR, and saves the extracted text to a JSON file.

To run the script, use the following command:

```bash
python apply_ocr.py debug
```

The `debug` argument is optional. If you include the `debug` argument, the script will save the images processed and used by the OCR to the `./debug` directory.
