import cv2
import pytesseract
import os
import numpy as np
import json
import re

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'
pokemon_data_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swsh/pokemon_data.json'

def _getframe(vid: cv2.VideoCapture):
    _, frame = vid.read()
    # cv2.imshow('game', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise SystemExit(0)
    return frame

def extract_text_from_box_coords(image_path, box):
    image = cv2.imread(image_path)
    x1, x2, y1, y2 = box
    x, y, w, h = x1, y1, x2 - x1, y2 - y1
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return pytesseract.image_to_string(thresh, config='--psm 7').strip()

# def get_swap_info():
#     regions = {
#         "p1Name": (735, 935, 120, 160),
#         "p2Name": (735, 935, 160, 200),
#         "hp": (900, 1080, 200, 230),
#         "attack": (735, 915, 240, 270),
#         "defense": (735, 915, 270, 300),
#         "sp_atk": (900, 1080, 240, 270),
#         "sp_def": (900, 1080, 270, 300),
#         "speed": (900, 1080, 300, 330)
#     }

#     data = {}
#     for key, box in regions.items():
#         text = extract_text_from_box_coords(image, box)
#         data[key] = text

#     return data

def extract_text(image_path, text_color="white", x1=None, y1=None, x2=None, y2=None, sortByX=False):
    """
    Extracts text from an image, filtering only Pokémon types.
    
    Parameters:
    - image_path (str): Path to the image.
    - pokemon_data_path (str): Path to the Pokémon types JSON file.
    - text_color (str): 'white' for white text, 'black' for black text.
    - x1, y1, x2, y2 (int, optional): Bounding box for text filtering.
    - sortByX (bool): Sort detected text by X-coordinate if True, else by Y-coordinate.

    Returns:
    - list: Extracted Pokémon type names in sorted order.
    """

    # Read the image and convert to HSV
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if text_color=="white" else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define color range for text detection
    if text_color.lower() == "white":
        lower_bound, upper_bound = np.array([0, 0, 200], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8)
    elif text_color.lower() == "black":
        lower_bound, upper_bound = np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 50], dtype=np.uint8)
    else:
        raise ValueError("Invalid text_color. Use 'white' or 'black'.")

    # Create a mask for the selected text color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # OCR settings
    custom_config = r'--oem 3 --psm 11'
    boxes = pytesseract.image_to_data(mask, config=custom_config, output_type=pytesseract.Output.DICT)

    text_data = []
    for i in range(len(boxes['text'])):
        text = boxes['text'][i].strip().lower()  # Convert to lowercase
        if len(text) > 1:  # Ignore short noise
            x, y = boxes['left'][i], boxes['top'][i]
            if x1 is None or y1 is None or x2 is None or y2 is None or (x1 <= x <= x2 and y1 <= y <= y2):
                text_data.append((x if sortByX else y, text))

    # Sort extracted text
    text_data.sort()
    sorted_text = [text for _, text in text_data]

    # Load Pokémon types from JSON
    with open(pokemon_data_path, 'r') as file:
        data = json.load(file)

    # Filter text to keep only valid Pokémon types
    # sorted_text = [text for text in sorted_text if text in data['pokemon_types']]

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

image_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swap.png'
# x1 = 174
# y1 = 62
# x2 = 1278
# y2 = 567

# Items
# x1 = 702
# y1 = 66
# x2 = 1261
# y2 = 433

# Swapping pokemon
# x1 = 585
# y1 = 103
# x2 = 1263
# y2 = 528

# Pokemon 1 types
x1 = 740
y1 = 148
x2 = 848
y2 = 221

# Example Usage
arrow_position = extract_text_from_box_coords(image_path=image_path,box= (x1, x2, y1, y2))
# arrow_position = extract_swap_string(image_path, x1 = x1, y1 = y1, x2 = x2, y2 = y2, )
print(arrow_position)
# print(extract_text(image_path, x1 = x1, y1 = y1, x2 = x2, y2 = y2))
