import cv2
import pytesseract
import os

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
# Set path to Tesseract executable (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'

def draw_text_boxes(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to get bounding box data
    boxes = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    
    # Loop over each detected word
    for i in range(len(boxes['text'])):
        if boxes['text'][i].strip():  # Ignore empty strings
            x, y, w, h = (boxes['left'][i], boxes['top'][i], 
                          boxes['width'][i], boxes['height'][i])
            
            # Draw rectangle around word
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put the detected text above the rectangle
            cv2.putText(image, boxes['text'][i], (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show the final image with bounding boxes
    cv2.imshow('Text Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
draw_text_boxes('/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/screenshot.png')