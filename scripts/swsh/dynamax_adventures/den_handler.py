import time
from utils import Point, _press, _getframe, get_text, increment_counter, check_if_shiny
import cv2
import serial
import pytesseract
import os
import time
import json
import numpy as np
from config_manager import ConfigManager

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'
pokemon_data_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swsh/pokemon_data.json'

currently_hunting = 'Articuno'

class DenHandler:
    def __init__(self, vid: cv2.VideoCapture, ser: serial.Serial, config: ConfigManager):
        self.vid = vid
        self.ser = ser
        self.config = config
    
    def handle_catch(self, is_legend:bool):
        print('Catching')
        _press(self.ser, 'A', sleep_time=1)

        if is_legend:
            ball_index = self.config.get('ball_index')
            _press(self.ser, 'a', sleep_time=0.5, count=ball_index)

        _press(self.ser, 'A', sleep_time=1)

    def select_starter(self): 
        if (self.config.get('selected_starter')):
            return
        print('Selecting starter!')
        _press(self.ser, '+', sleep_time=1.5)
        frame = _getframe(self.vid)
        try: first_attack = int(get_text(frame=frame, top_left=Point(y=175, x=971), bottom_right=Point(y=207, x=1041), invert=True))
        except: first_attack = 0
        try: first_specattack = int(get_text(frame=frame, top_left=Point(y=138, x=1191), bottom_right=Point(y=171, x=1250), invert=True))
        except: first_specattack = 0

        try: second_attack = int(get_text(frame=frame, top_left=Point(y=361, x=971), bottom_right=Point(y=394, x=1043), invert=True))
        except: second_attack = 0
        try: second_specattack = int(get_text(frame=frame, top_left=Point(y=323, x=1190), bottom_right=Point(y=356, x=1250), invert=True))
        except: second_specattack = 0

        try: third_attack = int(get_text(frame=frame, top_left=Point(y=547, x=969), bottom_right=Point(y=582, x=1050), invert=True))
        except: third_attack = 0
        try: third_specattack = int(get_text(frame=frame, top_left=Point(y=508, x=1193), bottom_right=Point(y=543, x=1250), invert=True))
        except: third_specattack = 0

        first_largest = (max(first_attack, first_specattack), 0)
        second_largest = (max(second_attack, second_specattack), 0)
        third_largest = (max(third_attack, third_specattack), 0)

        values = [first_largest, second_largest, third_largest]
        sorted_list = sorted(values, key=lambda x: x[0], reverse=True)

        distance = sorted_list[0][1]
        _press(self.ser, 's', count=distance, sleep_time=0.3)
        _press(self.ser, 'A')
        self.config.update({'selected_starter': True})

    def handle_choose_pokemon(self):
        print("Choosing")
        index = 0
        name_map = {}

        _press(self.ser, 'A', sleep_time=1.5)
        _press(self.ser, 's', sleep_time=1)
        _press(self.ser, 'A', sleep_time=4)
        
        frame = _getframe(self.vid)
        log_frames = []
        current_name = get_text(frame=frame, top_left=Point(y=87, x=279), bottom_right=Point(y=127, x=595), invert=True)
        while not any(value[0] == current_name for value in name_map.values()):
            pokemon_is_shiny = check_if_shiny(self.vid)
            print(f'Checking pokemon {index}, name: {current_name}, is shiny: {pokemon_is_shiny}')
            name_map[index] = (current_name, pokemon_is_shiny)
            log_frames.append(frame)
            index += 1
            _press(self.ser, 's')
            time.sleep(3)
            frame = _getframe(self.vid)
            current_name = get_text(frame=frame, top_left=Point(y=87, x=279), bottom_right=Point(y=127, x=595), invert=True)

        contains_legendary = name_map.__len__() == 4
        print(f'Legendary: {contains_legendary}')
        print(f'We have processed all pokemon: {name_map}')
        _press(self.ser, 'B', sleep_time=4)
        increment_counter(currently_hunting, frames=log_frames, caught_legend=contains_legendary)
        last_key, last_value = next(reversed(name_map.items()))
        first_true_key = next((key for key, (_, flag) in name_map.items() if flag), None)
        self.handle_den_search(contains_shiny = first_true_key is not None, beat_legend=contains_legendary)
        self.config.update({'move_index': 0, 'battle_index': 0,'dynamax_turns': None, 'selected_starter': False})
        if (contains_legendary and last_value[1]):
            print(f'Shiny legendary at index: {last_key}')
            self.clear_streak_data()
            _press(self.ser, 's', count=3, sleep_time=0.5)
            return True
        


        if (self.config.get('end_run')):
            return True

        if (first_true_key is None):
            print('Not taking any pokemon')
            _press(self.ser, 'B', sleep_time=1)
            _press(self.ser, 'A', sleep_time=1, count=3)
            return False
            
        print(f'Take according pokemon {first_true_key}')
        _press(self.ser, 's', count=first_true_key)
        self.take_pokemon()
        return False
    
    def handle_den_search(self, contains_shiny: bool, beat_legend:bool):
        streak_data = self.config.get('streak_data')
        if (streak_data['search_dens'] == False):
            streak_data['paths'] = []
            streak_data['swaps'] = []
            self.config.update({"streak_data":streak_data,})
            return
        
        print('Handling den search')
        if (beat_legend):
            streak_data['wins'] = streak_data['wins'] + 1
        
        streak_data['battles'] = streak_data['battles'] + 1
        
        streak_percent = streak_data['wins'] / streak_data['battles']

        keep_dungeon = False

        if (streak_data['wins'] > 0 and streak_data['battles'] < 2 or streak_percent >= streak_data['win_ratio_threshold']):
            keep_dungeon = True

        if (contains_shiny and streak_data['battles'] < 4 and streak_percent != 1):
            keep_dungeon = False

        if (keep_dungeon == False):
            self.clear_streak_data()
            return

        self.config.update({"streak_data":streak_data, "keep_dungeon":keep_dungeon})

    def clear_streak_data(self):
        print("Clearing streak data")
        streak_data = self.config.get('streak_data')
        streak_data['battles'] = 0
        streak_data['wins'] = 0 
        streak_data['paths'] = []
        streak_data['swaps'] = []

        self.config.update({"streak_data":streak_data, "keep_dungeon": False})

    def take_pokemon(self):
        _press(self.ser, 'A', sleep_time=3.5, count=5)
        _press(self.ser, 'A', sleep_time=1)

    def swap_if_needed(self):
        print('Swapping')
        _press(self.ser, '+', sleep_time=1.5)
        frame = _getframe(self.vid)

        streak_data = self.config.get('streak_data')
        if (len(streak_data['swaps']) == 3 and streak_data['search_dens'] == True):
            would_swap = streak_data['swaps'][self.config.get('battle_index') - 1]
            print(f'Swapping based on streak: {would_swap}')
            if (would_swap):
                print(f'Swaping')
                _press(self.ser, 'A')
            else:
                print('Keeping current')
                _press(self.ser, 'B')
            return

        try: curr_attack = int(get_text(frame=frame, top_left=Point(y=211, x=969), bottom_right=Point(y=247, x=1056), invert=True))
        except: curr_attack = 0
        try: curr_specattack = int(get_text(frame=frame, top_left=Point(y=178, x=1192), bottom_right=Point(y=210, x=1249), invert=True))
        except: curr_specattack = 0

        try: temp_attack = int(get_text(frame=frame, top_left=Point(y=398, x=971), bottom_right=Point(y=435, x=1049), invert=True))
        except: temp_attack = 0
        try: temp_specattack = int(get_text(frame=frame, top_left=Point(y=359, x=1192), bottom_right=Point(y=399, x=1253), invert=True))
        except: temp_specattack = 0

        curr_max = max(curr_attack, curr_specattack)
        temp_max = max(temp_attack, temp_specattack)

        would_swap = temp_max > curr_max

        if (would_swap):
            print(f'Swaping')
            _press(self.ser, 'A')
        else:
            print('Keeping current')
            _press(self.ser, 'B')

        streak_data['swaps'].append(would_swap)
        self.config.update({"streak_data":streak_data})

    def extract_text(self, x1, y1, x2, y2, sortByX: bool):
        image = _getframe(self.vid)
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
                    text_data.append((x if sortByX else y, text.lower()))
        
        text_data.sort()
        sorted_text = [text for _, text in text_data]

        with open(pokemon_data_path, 'r') as file:
            data = json.load(file)
        
        sorted_text = [text for text in sorted_text if text in data['pokemon_types']]

        return sorted_text
    
    def handle_select_item(self):
        print('Selecting item')
        _press(self.ser, 'A')
    
    def handle_rental(self):
        print('Rental pokemon')
        _press(self.ser, 'B')
    
    def handle_sus(self):
        print('Sus screen')
        _press(self.ser, 'A', sleep_time=5, count=2)

    pathX1 = 174
    pathY1 = 62
    pathX2 = 1278
    pathY2 = 567

    def find_white_arrow(self):
        image = _getframe(self.vid)
        cropped_image = image[self.pathY1:self.pathY2, self.pathX1:self.pathX2]
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, _, _, _ = cv2.boundingRect(largest_contour)
            return x+self.pathX1

        return None 
    
    def choose_best_index(self, values):
        type_order = self.config.get('type_order')

        if (len(values) == 0):
            return 0

        first_match = min(values, key=lambda x: type_order.index(x))

        # Get its index in the `others` list
        index_of_path = values.index(first_match)

        return index_of_path 
    
    def handle_pathing(self):
        print('Pathing')
        types = self.extract_text(self.pathX1, self.pathY1, self.pathX2, self.pathY2, sortByX=True)
        
        # If we are on a streak make sure to take same path
        streak_data = self.config.get('streak_data')
        if (len(streak_data['paths']) == 3 and streak_data['search_dens'] == True):
            index = streak_data['paths'][self.config.get('battle_index')]
            print(f'Taking path based on streak: {index}')
            _press(self.ser, 'd', count=index, sleep_time=0.3)
            _press(self.ser, 'A')
            return


        first_path = types.__len__() == 2

        arrow_positions = []
        new_pos = self.find_white_arrow()

        while not any(new_pos < pos + 50 and new_pos > pos - 50 for pos in arrow_positions):
            print(f'Adding arrow position: {new_pos}')
            arrow_positions.append(new_pos)
            _press(self.ser, 'd', sleep_time=0.75)
            new_pos = self.find_white_arrow()

        print(f'Arrow positions currently: {arrow_positions}')

        if arrow_positions[0] < 350 or first_path:
            type_index = 0
        elif arrow_positions.__len__() == 3:
            type_index = 1
        else:
            type_index = 2
        
        print(f'Arrow is currently on pokemon: {type_index}')

        types = types[type_index:(type_index + min(arrow_positions.__len__(), types.__len__() ))]

        print(f'Current path types available: {types}')
        best_index = self.choose_best_index(types)
        streak_data['paths'].append(best_index)
        self.config.update({"streak_data":streak_data})
        print(f'Taking path at index {best_index}')
        _press(self.ser, 'd', count=best_index, sleep_time=0.3)
        _press(self.ser, 'A')

    def restart_dungeon(self, keep_dungeon = False):
        if (keep_dungeon):
            self.reset_game()

        frame = _getframe(self.vid)
        curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)
        while curr_text != 'Dynamax Adventure?':
            _press(self.ser, 'A')
            time.sleep(2)
            frame = _getframe(self.vid)
            curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)
        
        pokemon_den_index = self.config.get('pokemon_den_index')
        # Would you like to embark on a Dynamax Adventure?
        _press(self.ser, 'A', sleep_time=2, count=4)
        _press(self.ser, 's', sleep_time=0.5, count=pokemon_den_index)
        _press(self.ser, 'A', sleep_time=2, count=3)
        time.sleep(4)

        # Dont invite others
        _press(self.ser, 's', sleep_time=0.5)
        _press(self.ser, 'A')

    def reset_game(self):
        _press(self.ser, 'H', sleep_time=1)
        _press(self.ser, 'X', sleep_time=1)
        _press(self.ser, 'A', sleep_time=1, count=3)

        print('game reset!')