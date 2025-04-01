import cv2
import pytesseract
import os
import numpy as np
import json

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
# Set path to Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'

pokemon_data_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swsh/pokemon_data.json'

def _getframe(vid: cv2.VideoCapture):
    _, frame = vid.read()
    # cv2.imshow('game', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise SystemExit(0)
    return frame

def extract_text(image_path, x1=None, y1=None, x2=None, y2=None):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    custom_config = r'--oem 3 --psm 11'
    boxes = pytesseract.image_to_data(mask, config=custom_config, output_type=pytesseract.Output.DICT)
    
    text_data = []
    for i in range(len(boxes['text'])):
        text = boxes['text'][i].strip()
        if len(text) > 1:  # Ignore very short words (likely noise)
            x, y, w, h = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            if x1 is None or y1 is None or x2 is None or y2 is None or (x1 <= x <= x2 and y1 <= y <= y2):
                text_data.append((x, text.lower()))
    
    text_data.sort()
    sorted_text = [text for _, text in text_data]

    with open(pokemon_data_path, 'r') as file:
        data = json.load(file)
    
    sorted_text = [text for text in sorted_text if text in data['pokemon_types']]

    return sorted_text

def find_white_arrow(image_path, x1, y1, x2, y2):
    image = cv2.imread(image_path)
    cropped_image = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x+x1

    return None 

image_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/path3.png'
x1 = 174
y1 = 62
x2 = 1278
y2 = 567

# Example Usage
arrow_position = find_white_arrow(image_path, x1 = x1, y1 = y1, x2 = x2, y2 = y2)
print(arrow_position)
# print(extract_text(image_path, x1 = x1, y1 = y1, x2 = x2, y2 = y2))
