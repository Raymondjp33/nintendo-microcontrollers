from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import random
import time
import json
from collections.abc import Generator
from typing import NamedTuple
import tesserocr
from typing import Protocol
import functools
import numpy as np

import cv2
import numpy
import serial
import pytesseract
import os

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'
pokemon_data_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swsh/pokemon_data.json'


class Color(NamedTuple):
    b: int
    g: int
    r: int
class Point(NamedTuple):
    y: int
    x: int

    def norm(self, dims: tuple[int, int, int]) -> Point:
        return type(self)(
            int(self.y / NORM.y * dims[0]),
            int(self.x / NORM.x * dims[1]),
        )

    def denorm(self, dims: tuple[int, int, int]) -> Point:
        return type(self)(
            int(self.y / dims[0] * NORM.y),
            int(self.x / dims[1] * NORM.x),
        )

NORM = Point(y=720, x=1280)

def _press(ser: serial.Serial, s: str, duration: float = .1, count: int = 1, sleep_time = .075) -> None:
    for _ in range(count):
        # print(f'{s=} {duration=}')
        ser.write(s.encode())
        time.sleep(duration)
        ser.write(b'0')
        time.sleep(sleep_time)

def _getframe(vid: cv2.VideoCapture) -> numpy.ndarray:
    _, frame = vid.read()
    # cv2.imshow('game', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise SystemExit(0)
    return frame

def _wait_and_render(vid: cv2.VideoCapture, t: float) -> None:
    end = time.time() + t
    while time.time() < end:
        _getframe(vid)

def _await_pixel(
        ser: serial.Serial,
        vid: cv2.VideoCapture,
        *,
        x: int,
        y: int,
        pixel: tuple[int, int, int],
        timeout: float = 90,
        exact_pixel: bool = True,
) -> None:
    end = time.time() + timeout
    frame = _getframe(vid)

    compare = numpy.array_equal if exact_pixel else _color_near

    while not compare(frame[y][x], pixel):
        frame = _getframe(vid)

def _await_not_pixel(
        ser: serial.Serial,
        vid: cv2.VideoCapture,
        *,
        x: int,
        y: int,
        pixel: tuple[int, int, int],
        timeout: float = 90,
        exact_pixel: bool = True,
) -> None:
    end = time.time() + timeout
    frame = _getframe(vid)
    compare = numpy.array_equal if exact_pixel else _color_near
    while compare(frame[y][x], pixel):
        frame = _getframe(vid)

def _color_near(pixel: numpy.ndarray, expected: tuple[int, int, int]) -> bool:
    total = 0
    for c1, c2 in zip(pixel, expected):
        total += (c2 - c1) * (c2 - c1)

    return total < 76

@contextlib.contextmanager
def _shh(ser: serial.Serial) -> Generator[None]:
    try:
        yield
    finally:
        ser.write(b'.')    

@functools.lru_cache
def _tessapi() -> tesserocr.PyTessBaseAPI:
    return tesserocr.PyTessBaseAPI(
        #/opt/homebrew/Cellar/tesseract/5.5.0/share/
        '/opt/homebrew/share/tessdata',
        'eng',
        psm=tesserocr.PSM.SINGLE_LINE,
    )

class Matcher(Protocol):
    def __call__(self, frame: numpy.ndarray) -> bool: ...

def tess_text_u8(
        img: numpy.ndarray,
        *,
        tessapi: tesserocr.PyTessBaseAPI | None = None,
) -> str:
    tessapi = tessapi or _tessapi()

    tessapi.SetImageBytes(
        img.tobytes(),
        width=img.shape[1],
        height=img.shape[0],
        bytes_per_pixel=1,
        bytes_per_line=img.shape[1],
    )
    return tessapi.GetUTF8Text().strip()

def get_text(
        frame: numpy.ndarray,
        top_left: Point,
        bottom_right: Point,
        *,
        invert: bool,
        tessapi: tesserocr.PyTessBaseAPI | None = None,
) -> str:
    tl_norm = top_left.norm(frame.shape)
    br_norm = bottom_right.norm(frame.shape)

    crop = frame[tl_norm.y:br_norm.y, tl_norm.x:br_norm.x]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, crop = cv2.threshold(
        crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    if invert:
        crop = cv2.bitwise_not(crop)

    return tess_text_u8(crop, tessapi=tessapi)

def match_text(
        text: str,
        top_left: Point,
        bottom_right: Point,
        *,
        invert: bool,
) -> Matcher:
    def match_text_impl(frame: numpy.ndarray) -> bool:
        print(f'here and text is {text} and the gotten text is ${get_text(frame, top_left, bottom_right, invert=invert)}')
        return text == get_text(frame, top_left, bottom_right, invert=invert)
    return match_text_impl

def increment_counter(file_prefix, frames=None, caught_legend=False):
    total_dens_counter_path = Path(f'{file_prefix}-total-dens-counter.txt')
    counter_path = Path(f'{file_prefix}-counter.txt')
    # log_path = Path(f"{file_prefix}-log.txt")
    
    # Read the existing count (default to 0 if file does not exist)
    if counter_path.exists():
        with counter_path.open("r") as file:
            try:
                count = int(file.read().strip())
            except ValueError:
                count = 0
    else:
        count = 0

    if total_dens_counter_path.exists():
        with total_dens_counter_path.open("r") as file:
            try:
                total_dens_count = int(file.read().strip())
            except ValueError:
                total_dens_count = 0
    else:
        total_dens_count = 0

    total_dens_count += 1

    # Increment the counter
    if caught_legend:
        count += 1

    # timestamp  = time.strftime('%Y-%m-%d %H:%M:%S')
    # star = '* ' if delay > 0.53 or delay < 0.47 else ''
    # log_data = f'{star}Count: {count} - Delay: {delay} - Timestamp {timestamp}'

    # Write the updated count back to the file
    with counter_path.open("w") as file1, total_dens_counter_path.open("w") as file2:
        file1.write(str(count))
        file2.write(str(total_dens_count))
        # file2.write(log_data + '\n')
    
    if frames is not None:
        for x in range(frames.__len__()):
            cv2.imwrite(f"/Volumes/Untitled/dynamax/{total_dens_count}-{f'{file_prefix}-{count}' if x == 3 else x}.png", frames[x])
  
def write_shiny_text():
    shiny_text_path = Path(f"shiny_text.txt")
    with shiny_text_path.open("w") as file1:
        file1.write('I got the shiny! My switch\nwill be off until I am\nback. Make sure to come\nback when/after I catch it!')

def connect_and_go_to_game(ser: serial.Serial):
    _press(ser, 'H', sleep_time=1)
    _press(ser, 'H', duration=0.1)
    _press(ser, 'A', duration=0.5)
    _press(ser, 'H', duration=0.1)
    _press(ser, 'A', sleep_time=1)
    _press(ser, 'A')
    _press(ser, '0')

def go_to_change_grip(ser: serial.Serial):
    _press(ser, 'H')
    time.sleep(1)
    _press(ser, 's')
    _press(ser, 'd', count=4)
    _press(ser, 'A')
    time.sleep(1)
    _press(ser, 'A')
    
def reset_game(ser: serial.Serial, vid: cv2.VideoCapture,):
    _press(ser, 'H', sleep_time=1)
    _press(ser, 'X', sleep_time=1)
    _press(ser, 'A', sleep_time=1, count=3)

    print('game reset!')

# E for effectivness and P for PP
class Move(NamedTuple):
    e: str
    p: int
    index: int

move_index = 0
dynamax_turns = None
battle_index = 0
def attack_with_move(vid: cv2.VideoCapture, ser: serial.Serial):
    global move_index
    global dynamax_turns
    move1ETL = Point(y=471, x=919)
    move1EBR = Point(y=493, x=1050)
    move1PTL = Point(y=445, x=1154)
    move1PBR = Point(y=487, x=1252)

    move2ETL = Point(y=541, x=919)
    move2EBR = Point(y=561, x=1050)
    move2PTL = Point(y=512, x=1154)
    move2PBR = Point(y=557, x=1252)

    move3ETL = Point(y=608, x=919)
    move3EBR = Point(y=632, x=1050)
    move3PTL = Point(y=581, x=1154)
    move3PBR = Point(y=626, x=1252)

    move4ETL = Point(y=678, x=919)
    move4EBR = Point(y=702, x=1050)
    move4PTL = Point(y=649, x=1154)
    move4PBR = Point(y=698, x=1252)

    moves_order = {
        "Super effective": 0,
        "Effective": 1,
        "Not very effective": 2
    }

    def sort_key(move: Move):
        # Moves with count == 0 should go last
        count_is_zero = 1 if move.p == 0 else 0
        # Moves not in the predefined order should go last
        move_rank = moves_order.get(move.e, 3)  # Unrecognized moves get a high value (3)
        return (count_is_zero, move_rank)

    frame = _getframe(vid)
    try:
        move1E = get_text(frame=frame, top_left=move1ETL, bottom_right=move1EBR, invert=True)

        move1P = int(get_text(frame=frame, top_left=move1PTL, bottom_right=move1PBR, invert=True).split('/')[0])
    except:
        move1P = 0

    move2E = get_text(frame=frame, top_left=move2ETL, bottom_right=move2EBR, invert=True)
    try:
        move2P = int(get_text(frame=frame, top_left=move2PTL, bottom_right=move2PBR, invert=True).split('/')[0])
    except:
        move2P = 0

    move3E = get_text(frame=frame, top_left=move3ETL, bottom_right=move3EBR, invert=True)
    try:
        move3P = int(get_text(frame=frame, top_left=move3PTL, bottom_right=move3PBR, invert=True).split('/')[0])
    except:
        move3P = 0

    move4E = get_text(frame=frame, top_left=move4ETL, bottom_right=move4EBR, invert=True)
    try:
        move4P = int(get_text(frame=frame, top_left=move4PTL, bottom_right=move4PBR, invert=False).split('/')[0])
    except:
        move4P = 0
    
    move_list = [Move(e=move1E, p=move1P, index=0), Move(e=move2E, p=move2P, index=1), Move(e=move3E, p=move3P, index=2), Move(e=move4E, p=move4P, index=3),]

    sorted_data = sorted(move_list, key=sort_key)

    if (dynamax_turns is not None):
        dynamax_turns -= 1

    if (dynamax_turns is not None and dynamax_turns < 0):
        dynamax_turns = None
        move_index = 0

    print(f'move1: {move1E} {move1P}\nmove2: {move2E} {move2P}\nmove3: {move3E} {move3P}\nmove4: {move4E} {move4P}')
    new_move_index = sorted_data[0].index
    print(f'About to use move: {new_move_index}\nCurrently at move {move_index}')
    distance = move_index - new_move_index

    _press(ser, 's' if distance < 0 else 'w', count=abs(distance), sleep_time=0.2)
    _press(ser, 'A', sleep_time=1)
    _press(ser, 'A', sleep_time=0.5)
    
    # Attempt using move on self if using it in general failed
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A', sleep_time=0.5)
    move_index = new_move_index

def dynamax_if_available(vid: cv2.VideoCapture, ser: serial.Serial):
    global dynamax_turns
    frame = _getframe(vid)

    dynamax_available = False

    for _ in range(10):
        if (_color_near(frame[590][728], (79, 0, 222))):
            dynamax_available = True
        time.sleep(0.1)
        frame = _getframe(vid)
    # print(f'Here! {frame[590][728]}')
    if (dynamax_available):
        print('Dynamaxing!')
        dynamax_turns = 3
        _press(ser, 'a', sleep_time=0.5)
        _press(ser, 'A', sleep_time=0.5)
    else:
        print('Not dynamaxing!')

def handle_choose_pokemon(vid: cv2.VideoCapture, ser: serial.Serial, end_run = False):
    print("Choosing")
    index = 0
    name_map = {}

    _press(ser, 'A', sleep_time=1.5)
    _press(ser, 's', sleep_time=1)
    _press(ser, 'A', sleep_time=4)
    
    frame = _getframe(vid)
    log_frames = []
    current_name = get_text(frame=frame, top_left=Point(y=87, x=279), bottom_right=Point(y=127, x=595), invert=True)
    log_frame = None
    while not any(value[0] == current_name for value in name_map.values()):
        pokemon_is_shiny = check_if_shiny(vid)
        print(f'Checking pokemon {index}, name: {current_name}, is shiny: {pokemon_is_shiny}')
        name_map[index] = (current_name, pokemon_is_shiny)
        log_frames.append(frame)
        index += 1
        _press(ser, 's')
        time.sleep(3)
        frame = _getframe(vid)
        current_name = get_text(frame=frame, top_left=Point(y=87, x=279), bottom_right=Point(y=127, x=595), invert=True)

    contains_legendary = name_map.__len__() == 4
    print(f'Legendary: {contains_legendary}')
    print(f'We have processed all pokemon: {name_map}')
    _press(ser, 'B', sleep_time=4)
    increment_counter('Reshiram', frames=log_frames, caught_legend=contains_legendary)
    last_key, last_value = next(reversed(name_map.items()))
    if (contains_legendary and last_value[1]):
        print(f'Shiny legendary at index: {last_key}')
        return True
    
    first_true_key = next((key for key, (_, flag) in name_map.items() if flag), None)

    if (end_run):
        return True

    if (first_true_key is None):
        print('Not taking any pokemon')
        _press(ser, 'B', sleep_time=1)
        _press(ser, 'A', sleep_time=1, count=3)
        return False
        
    print(f'Take according pokemon {first_true_key}')
    _press(ser, 's', count=first_true_key)
    take_pokemon(ser)
    return False

def take_pokemon(ser: serial.Serial):
    _press(ser, 'A', sleep_time=1.5, count=5)
    time.sleep(2)
    _press(ser, 'A', sleep_time=1)

def swap_if_needed(vid: cv2.VideoCapture, ser: serial.Serial):
    print('Swapping')
    _press(ser, '+', sleep_time=1.5)
    frame = _getframe(vid)

    try: curr_attack = int(get_text(frame=frame, top_left=Point(y=211, x=969), bottom_right=Point(y=247, x=1056), invert=True))
    except: curr_attack = 0
    try: curr_specattack = int(get_text(frame=frame, top_left=Point(y=178, x=1192), bottom_right=Point(y=210, x=1249), invert=True))
    except: curr_specattack = 0

    try: temp_attack = int(get_text(frame=frame, top_left=Point(y=401, x=971), bottom_right=Point(y=432, x=1049), invert=True))
    except: temp_attack = 0
    try: temp_specattack = int(get_text(frame=frame, top_left=Point(y=362, x=1195), bottom_right=Point(y=394, x=1249), invert=True))
    except: temp_specattack = 0

    curr_max = max(curr_attack, curr_specattack)
    temp_max = max(temp_attack, temp_specattack)

    would_swap = temp_max > curr_max

    if (would_swap):
        print(f'Swaping')
        _press(ser, 'A')
    else:
        print('Keeping current')
        _press(ser, 'B')

selected = False
def select_starter(vid: cv2.VideoCapture, ser: serial.Serial):
    global selected
    if (selected):
        return
    print('Selecting starter!')
    _press(ser, '+', sleep_time=1.5)
    frame = _getframe(vid)
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
    _press(ser, 's', count=distance, sleep_time=0.3)
    _press(ser, 'A')
    selected = True

def handle_fight(vid: cv2.VideoCapture, ser: serial.Serial):
    print('Fighting')
    _press(ser, 'A', sleep_time=3)

    dynamax_if_available(vid, ser)
    attack_with_move(vid, ser)
    
def handle_catch(ser: serial.Serial, is_legend:bool):
    print('Catching')
    _press(ser, 'A', sleep_time=1)

    if is_legend:
        _press(ser, 'a', sleep_time=0.5, count=3)

    _press(ser, 'A', sleep_time=1)

def get_screen(vid: cv2.VideoCapture):
    frame = _getframe(vid)

    if (get_text(frame=frame, top_left=Point(y=502, x=1056), bottom_right=Point(y=539, x=1133), invert=True) == 'Fight'):
        return 'Fight'
    
    if (get_text(frame=frame, top_left=Point(y=46, x=26), bottom_right=Point(y=86, x=298), invert=True) == "One Trainer can choose"):
        return 'Swapping'
    
    if (get_text(frame=frame, top_left=Point(y=163, x=691), bottom_right=Point(y=198, x=1223), invert=True) == "Choose the one Pokémon you'd like to keep!"):
        return 'Choosing'
    
    if (get_text(frame=frame, top_left=Point(y=609, x=1091), bottom_right=Point(y=643, x=1205), invert=True) == 'Catch'):
        return 'Catching'
    
    if (get_text(frame=frame, top_left=Point(y=46, x=21), bottom_right=Point(y=85, x=238), invert=True) == 'Everyone will take'):
        return 'Selecting'
    
    if (get_text(frame=frame, top_left=Point(y=500, x=1053), bottom_right=Point(y=541, x=1182), invert=True) == 'Cheer On'):
        return 'Cheer On'
    
    if (get_text(frame=frame, top_left=Point(y=590, x=565), bottom_right=Point(y=642, x=689), invert=True) == 'letdown'):
        return 'Let down'
    
    if (get_text(frame=frame, top_left=Point(y=587, x=356), bottom_right=Point(y=626, x=493), invert=True) == 'Which path'):
        return 'Pathing'
    
    if (get_text(frame=frame, top_left=Point(y=51, x=344), bottom_right=Point(y=105, x=454), invert=True) == 'hold one'):
        return 'Items'
    
    if (get_text(frame=frame, top_left=Point(y=51, x=344), bottom_right=Point(y=105, x=454), invert=True) == 'hold one'):
        return 'Items'

def check_if_shiny(vid: cv2.VideoCapture):
    frame = _getframe(vid)

    y1, x1, y2, x2 = 383, 74, 418, 260
    roi = frame[y1:y2, x1:x2]
    target_color = (99, 22, 255)

    found = any(_color_near(pixel, target_color) for row in roi for pixel in row)

    if found:
        return True
    
    return False

def handle_select_item(vid: cv2.VideoCapture, ser: serial.Serial):
    print('Selecting item')
    _press(ser, 'A')
    # extract_text(vid, 702, 66, 1261, 433)

def restart_dungeon(vid: cv2.VideoCapture, ser: serial.Serial, keep_dungeon = False):
    global move_index
    global selected
    global dynamax_turns
    global battle_index
    
    move_index = 0
    battle_index = 0
    selected = False
    dynamax_turns = None

    if (keep_dungeon):
        reset_game(ser, vid)

    frame = _getframe(vid)
    curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)
    while curr_text != 'Dynamax Adventure?':
        _press(ser, 'A')
        time.sleep(2)
        frame = _getframe(vid)
        curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)

    # Would you like to embark on a Dynamax Adventure?
    _press(ser, 'A', sleep_time=2, count=4)
    _press(ser, 's', sleep_time=0.5, count=1)
    _press(ser, 'A', sleep_time=2, count=3)
    time.sleep(4)

    # Dont invite others
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A')

pathX1 = 174
pathY1 = 62
pathX2 = 1278
pathY2 = 567

def extract_text(vid: cv2.VideoCapture, x1, y1, x2, y2, sortByX: bool):
    image = _getframe(vid)
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

def find_white_arrow(vid: cv2.VideoCapture):
    image = _getframe(vid)
    cropped_image = image[pathY1:pathY2, pathX1:pathX2]
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, _, _, _ = cv2.boundingRect(largest_contour)
        return x+pathX1

    return None 

def handle_pathing(vid: cv2.VideoCapture, ser: serial.Serial):
    print('Pathing')
    types = extract_text(vid, pathX1, pathY1, pathX2, pathY2, sortByX=True)

    first_path = types.__len__() == 2

    arrow_positions = []
    new_pos = find_white_arrow(vid)

    while not any(new_pos < pos + 50 and new_pos > pos - 50 for pos in arrow_positions):
        print(f'Adding arrow position: {new_pos}')
        arrow_positions.append(new_pos)
        _press(ser, 'd', sleep_time=0.75)
        new_pos = find_white_arrow(vid)

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
    best_index = choose_best_index(types)
    print(f'Taking path at index {best_index}')
    _press(ser, 'd', count=best_index, sleep_time=0.3)
    _press(ser, 'A')

def choose_best_index(values):
    first_priority = []
    second_priority = []
    third_priority = []

    with open(pokemon_data_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    seek = data['seekTypes']
    avoid = data['avoidTypes']

    for i, value in enumerate(values):
        if value in seek:
            first_priority.append((i, value))  
        elif value not in avoid:
            second_priority.append((i, value))
        else:
            third_priority.append((i, value))

    if first_priority:
        return first_priority[0][0] 
    elif second_priority:
        return second_priority[0][0]  
    elif third_priority:
        return third_priority[0][0]  

    return None  

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-120')
    args = parser.parse_args()

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    shiny_legend = False
    global move_index
    global selected
    global dynamax_turns
    global battle_index

    with serial.Serial(args.serial, 9600) as ser, _shh(ser):
        time.sleep(1)
        # go_to_change_grip(ser)
        # connect_and_go_to_game(ser)
        # handle_choose_pokemon()
        # take_pokemon(ser)
        # restart_dungeon(vid, ser)
        # select_starter(vid, ser)
        # handle_choose_pokemon(vid, ser)
        # return 0
        
        while not shiny_legend:

            screen = get_screen(vid)

            if (screen == 'Fight'):
                handle_fight(vid, ser)
            
            if (screen == 'Swapping'):
                swap_if_needed(vid, ser)

            if (screen == 'Catching'):
                move_index = 0
                dynamax_turns = None
                battle_index += 1
                handle_catch(ser, is_legend=battle_index==4)

            if (screen == 'Selecting'):
                select_starter(vid, ser)

            if (screen == 'Choosing'):
                shiny_legend = handle_choose_pokemon(vid, ser, end_run=False)

                if (shiny_legend):
                    break

                restart_dungeon(vid, ser, keep_dungeon=True)
                # return 0

            if (screen == 'Cheer On'):
                if (dynamax_turns is not None):
                    dynamax_turns = -1
                print('Cheering')
                _press(ser, 'A')

            if (screen == 'Let down'):
                print('Handling let down')
                restart_dungeon(vid, ser, keep_dungeon=False)

            if (screen == 'Pathing'):
                handle_pathing(vid, ser)

            if (screen == 'Items'):
                handle_select_item(vid, ser)

            time.sleep(5)

    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
