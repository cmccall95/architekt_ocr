import sys
import ocr
import json
import parsers.jacobs_table as jacob


def main():
    if len(sys.argv) < 2:
        raise "Provide at least one input image for OCR"

    output_boxes = ocr.apply_ocr(sys.argv[1])

    # At this time only JACOBS parsing is done so.
    print('PARSING JACOBS FORMAT')
    output = jacob.parse_jacobs(output_boxes)

    print('PARSED DONE --> OUTPUT IS SAVED IN output.json')
    with open('output.json', 'w') as out:
        out.write(json.dumps(output))


if __name__ == "__main__":
    main()
