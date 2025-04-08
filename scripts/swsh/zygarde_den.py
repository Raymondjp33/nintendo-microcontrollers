from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import random
import time
from collections.abc import Generator
from typing import NamedTuple
import tesserocr
from typing import Protocol
import functools
import numpy as np

import cv2
import numpy
import serial

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
    # time.sleep(sleep_time)
    # return
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
    counter_path = Path(f'{file_prefix}-counter.txt')
    total_dens_counter_path = Path(f'{file_prefix}-total-dens-counter.txt')
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
            cv2.imwrite(f"/Volumes/Untitled/dynamax/{file_prefix} - {count} - {x}.png", frames[x])
  
def write_shiny_text():
    shiny_text_path = Path(f"shiny_text_switch2.txt")
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
    _press(ser, 'H')
    time.sleep(1)
    _press(ser, 'X')
    time.sleep(1)
    _press(ser, 'A')
    time.sleep(3)

    frame = _getframe(vid)
    while not _color_near(frame[584][970], (255, 255, 255)):
        _press(ser, 'A')
        _wait_and_render(vid, .15)
        frame = _getframe(vid)

    print('game loaded!')

dynamax_turns = None
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

# E for effectivness and P for PP
class Move(NamedTuple):
    e: str
    p: int
    index: int

move_index = 0
def attack_with_move(vid: cv2.VideoCapture, ser: serial.Serial, battle_index: int):
    fighting_legend = battle_index == 3
    print(f'Fighting, and we are fighting Zygarde: {fighting_legend}')
    global move_index
    global dynamax_turns
    move1PTL = Point(y=445, x=1154)
    move1PBR = Point(y=487, x=1252)


    _press(ser, 'A', sleep_time=1.5)

    frame = _getframe(vid)
    try:
        move1P = int(get_text(frame=frame, top_left=move1PTL, bottom_right=move1PBR, invert=True).split('/')[0])
    except:
        move1P = 0
    
    if not fighting_legend:
        dynamax_if_available(vid, ser)

    if (dynamax_turns is not None):
        dynamax_turns -= 1

    if (dynamax_turns is not None and dynamax_turns < 0):
        dynamax_turns = None
        move_index = 0

    if battle_index == 0:
        new_move_index = 0
    if battle_index == 1:
        new_move_index = 2
    if battle_index == 2:
        new_move_index = 0
    if battle_index == 3:
        new_move_index = 1
    print(f'Currently on move: {move_index}. About to use move: {new_move_index}')
    distance = move_index - new_move_index

    _press(ser, 's' if distance < 0 else 'w', count=abs(distance), sleep_time=0.2)
    _press(ser, 'A', sleep_time=1)
    _press(ser, 'A')
    move_index = new_move_index

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
    print(f'We caught Zygarde: {contains_legendary}')
    print(f'We have processed all pokemon: {name_map}')
    _press(ser, 'B', sleep_time=4)
    if (contains_legendary):
        increment_counter('Zygarde', frames=[log_frames[-1]], caught_legend=contains_legendary)
    last_key, last_value = next(reversed(name_map.items()))
    if (contains_legendary and last_value[1]):
        print(f'We have the shiny!')
        return True

    if (end_run):
        return True
    
    return False

def swap_if_needed(ser: serial.Serial, needs_to_swap: bool):
    if (needs_to_swap):
        print(f'Swaping')
        _press(ser, 'A')
    else:
        print('Keeping current')
        _press(ser, 'B')

selected = False
def select_starter(ser: serial.Serial):
    global selected
    if (selected):
        return
    print('Selecting')
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A')
    selected = True
    
def handle_catch(ser: serial.Serial, is_legend:bool):
    print(f'Catching, is legend: {is_legend}')
    _press(ser, 'A', sleep_time=1)

    if is_legend:
        _press(ser, 'a', sleep_time=1)
    
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
    
    if (get_text(frame=frame, top_left=Point(y=587, x=356), bottom_right=Point(y=626, x=493), invert=True) == 'Which path'):
        return 'Pathing'

def check_if_shiny(vid: cv2.VideoCapture):
    frame = _getframe(vid)

    y1, x1, y2, x2 = 383, 74, 418, 260
    roi = frame[y1:y2, x1:x2]
    target_color = (99, 22, 255)

    found = any(_color_near(pixel, target_color) for row in roi for pixel in row)

    if found:
        return True
    
    return False

def restart_dungeon(vid: cv2.VideoCapture, ser: serial.Serial):
    print('Restarting game and dungeon!')
    reset_game(ser, vid)
    frame = _getframe(vid)
    curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)
    while curr_text != 'Dynamax Adventure?':
        _press(ser, 'A')
        time.sleep(2)
        frame = _getframe(vid)
        curr_text = get_text(frame=frame, top_left=Point(y=641, x=269), bottom_right=Point(y=690, x=593), invert=True)

    print('About to start new adventure!')

    # Would you like to embark on a Dynamax Adventure?
    _press(ser, 'A', sleep_time=2, count=4)
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A', sleep_time=2, count=3)
    time.sleep(4)

    # Dont invite others
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A')

def select_path(ser: serial.Serial, battle_index: int):
    path = 1 if battle_index == 0 else 1 if battle_index == 1 else 2

    print(f'Pathing and going over: {path}')


    _press(ser, 'd', sleep_time=0.5, count=path)
    _press(ser, 'A')

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-120')
    args = parser.parse_args()

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    shiny_legend = False
    battle_index = 0
    global move_index
    global selected
    global dynamax_turns
    count = 0

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
                attack_with_move(vid, ser, battle_index=battle_index)
            
            if (screen == 'Swapping'):
                swap_if_needed(ser, needs_to_swap=battle_index == 1)

            if (screen == 'Catching'):
                move_index = 0
                battle_index += 1
                dynamax_turns = None
                handle_catch(ser, is_legend=battle_index==4)

            if (screen == 'Selecting'):
                select_starter(ser)

            if (screen == 'Choosing'):
                move_index = 0
                battle_index = 0
                selected = False
                dynamax_turns = None
                shiny_legend = handle_choose_pokemon(vid, ser, end_run=False)

                if (shiny_legend):
                    # _press(ser, 'C', duration=2)
                    # _press(ser, 'H', duration=1)
                    # _press(ser, 's', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'd', duration=0.25)
                    # _press(ser, 'A', duration=1)
                    # _press(ser, 'w', duration=1)
                    # _press(ser, 'A', duration=1)
                    # write_shiny_text()
                    break
                

                restart_dungeon(vid, ser)
                # return 0

            if (screen == 'Cheer On'):
                print('Cheering')
                if (dynamax_turns is not None):
                    dynamax_turns = -1
                _press(ser, 'A')

            if (screen == 'Pathing'):
                select_path(ser, battle_index=battle_index)

            time.sleep(5)

    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
