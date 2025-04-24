from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from typing import NamedTuple
import tesserocr
from typing import Protocol
import functools
import numpy as np
from pathlib import Path

import cv2
import numpy
import serial
import pytesseract
import os

os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/Cellar/tesseract/5.5.0/share/tessdata'
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'

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

    # with open(pokemon_data_path, 'r') as file:
    #     data = json.load(file)
    
    # sorted_text = [text for text in sorted_text if text in data['pokemon_types']]

    return sorted_text

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

def check_if_shiny(vid: cv2.VideoCapture):
    frame = _getframe(vid)

    y1, x1, y2, x2 = 383, 74, 418, 260
    roi = frame[y1:y2, x1:x2]
    target_color = (99, 22, 255)

    found = any(_color_near(pixel, target_color) for row in roi for pixel in row)

    if found:
        return True
    
    return False

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
    
    if (get_text(frame=frame, top_left=Point(y=626, x=600), bottom_right=Point(y=667, x=674), invert=True) == 'rental'):
        return 'Rental'
    
    if (get_text(frame=frame, top_left=Point(y=242, x=245), bottom_right=Point(y=294, x=562), invert=True) == 'Play is being suspended.'):
        return 'Sus'
