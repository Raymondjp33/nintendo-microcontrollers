from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time
from collections.abc import Generator
from typing import NamedTuple
import functools
from typing import Protocol

import tesserocr

import cv2
import numpy
import serial

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

class Color(NamedTuple):
    b: int
    g: int
    r: int

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

@functools.lru_cache
def _tessapi() -> tesserocr.PyTessBaseAPI:
    return tesserocr.PyTessBaseAPI(
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

@contextlib.contextmanager
def _shh(ser: serial.Serial) -> Generator[None]:
    try:
        yield
    finally:
        ser.write(b'.') 

def connect_and_go_to_game(ser: serial.Serial):
    _press(ser, 'H')
    _press(ser, 'H', duration=0.1)
    _press(ser, 'A', duration=0.5)
    _press(ser, 'H', duration=0.1, sleep_time=0.5)
    _press(ser, 'A', sleep_time=0.5)
    _press(ser, '0')

def go_to_change_grip(ser: serial.Serial):
    _press(ser, 'H', sleep_time=0.75)
    _press(ser, 's')
    _press(ser, 'd', count=4)
    _press(ser, 'A')
    time.sleep(1)
    _press(ser, 'A')

# E for effectivness and P for PP
class Move(NamedTuple):
    e: int
    p: int
    index: int

def get_best_move(frame: numpy.ndarray,):
    move1E = get_text(frame=frame, top_left=Point(y=443, x=929), bottom_right=Point(y=460, x=1045), invert=True).split(' ').__len__()
    try:
        move1P = int(get_text(frame=frame, top_left=Point(y=414, x=1160), bottom_right=Point(y=456, x=1220), invert=True).split('/')[0])
    except:
        move1P = 0

    move2E = get_text(frame=frame, top_left=Point(y=518, x=929), bottom_right=Point(y=537, x=1046), invert=True).split(' ').__len__()
    try:
        move2P = int(get_text(frame=frame, top_left=Point(y=498, x=1160), bottom_right=Point(y=529, x=1220), invert=True).split('/')[0])
    except:
        move2P = 0

    move3E = get_text(frame=frame, top_left=Point(y=591, x=929), bottom_right=Point(y=613, x=1046), invert=True).split(' ').__len__()
    try:
        move3P = int(get_text(frame=frame, top_left=Point(y=564, x=1160), bottom_right=Point(y=605, x=1220), invert=True).split('/')[0])
    except:
        move3P = 0

    move4E = get_text(frame=frame, top_left=Point(y=664, x=929), bottom_right=Point(y=686, x=1046), invert=True).split(' ').__len__()
    try:
        move4P = int(get_text(frame=frame, top_left=Point(y=646, x=1157), bottom_right=Point(y=678, x=1225), invert=False).split('/')[0])
    except:
        move4P = 0
    
    move_list = [Move(e=move1E, p=move1P, index=0), Move(e=move2E, p=move2P, index=1), Move(e=move3E, p=move3P, index=2), Move(e=move4E, p=move4P, index=3),]
    order = {2: 0, 1: 1, 3: 2}

    sorted_moves = sorted(move_list, key=lambda m: (m.p == 0, order.get(m.e, float('inf'))))

    all_zero = all(m.p == 0 for m in move_list)
    if (all_zero):
        return -1

    # print(f'move1: {move1E} {move1P}, move2: {move2E} {move2P}, move3: {move3E} {move3P}, move4: {move4E} {move4P}')

    return sorted_moves[0].index

def move_to_grass(ser: serial.Serial):
    _press(ser, 's', duration=1)
    time.sleep(4)
    _press(ser, 'a', duration=1)
    _press(ser, 'w', duration=1)
    _press(ser, 'd', duration=4)
    _press(ser, 's', duration=1.15)
    _press(ser, 'd', duration=4.1)
    _press(ser, 'w', duration=0.30)

def move_to_center(ser: serial.Serial):
    _press(ser, 's', duration=0.30)
    _press(ser, 'a', duration=4)
    _press(ser, 'w', duration=1.15)
    _press(ser, 'a', duration=3.75)
    _press(ser, 's', duration=1)
    _press(ser, 'd', duration=1)
    _press(ser, 'w', duration=0.5)
    time.sleep(4)
    _press(ser, 'w', duration=1)

def center_heal_pokemon(ser: serial.Serial):
    _press(ser, 'A', sleep_time=0.5, count=10)
    time.sleep(4)
    _press(ser, 'B', sleep_time=0.5, count=5)

def save_game(ser: serial.Serial):
    _press(ser, 'X', sleep_time=1.5)
    _press(ser, 'R', sleep_time=3)
    _press(ser, 'A', sleep_time=1.5)
    time.sleep(4)

def check_move_learning(ser: serial.Serial, frame: numpy.ndarray):
    if (not _color_near(frame[50][90], (250, 250, 250))):
        return
    
    _press(ser, 's', sleep_time=0.5)
    _press(ser, 'A', sleep_time=1.5)
    _press(ser, 'B', sleep_time=1.5)

def go_heal_pokemon(ser: serial.Serial):
    _press(ser, 'a', duration=1)
    move_to_center(ser)
    center_heal_pokemon(ser=ser)
    move_to_grass(ser=ser)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-BG009FDF')
    args = parser.parse_args()

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    x_val = 960
    y_val = 660



  
    with serial.Serial(args.serial, 9600) as ser, _shh(ser):
        
        # save_game(ser=ser)
        # move_to_grass(ser=ser)
        # move_to_center(ser=ser)
        # center_heal_pokemon(ser=ser)
        # frame = _getframe(vid) 
        # print(get_best_move(frame=frame))
        # analyze_moves(frame=frame)
        go_to_change_grip(ser)
        # connect_and_go_to_game(ser)
        return 0
        while True:
            frame = _getframe(vid)
            go_right = False

            timeout = time.time() + 25
            while not _color_near(frame[50][90], (250, 250, 250)):
                _press(ser, 'd' if go_right else 'a', count=4)
                _wait_and_render(vid, .15)
                frame = _getframe(vid)
                go_right = not go_right

                if (timeout < time.time() and not _color_near(frame[50][90], (250, 250, 250))):
                    check_move_learning(ser=ser, frame=frame)
                    end = time.time() + 10
                    while time.time() < end and not _color_near(frame[50][90], (250, 250, 250)):
                      _press(ser, 'B', sleep_time=0.5)
                      frame = _getframe(vid)
                      check_move_learning(ser=ser, frame=frame)
                    timeout = time.time() + 25
                frame = _getframe(vid)

            print('Battle started!')
            time.sleep(2.5)

            frame = _getframe(vid)

            while not _color_near(frame[y_val][x_val], (255, 255, 255)):
                _wait_and_render(vid, .1)
                frame = _getframe(vid)
           
            print('You encountered a pokemon!')
            

            _await_not_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('pokemon pixel gone') 
            t0 = time.time()

            _await_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('Go Tiny!')
            t1 = time.time()

            delay = t1 - t0

            if (delay) > 0.7:
                print('SHINY!!!')
                _press(ser, 'C', duration=2)
                _press(ser, 'H', duration=1)
                _press(ser, 's', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'd', duration=0.25)
                _press(ser, 'A', duration=1)
                _press(ser, 'w', duration=1)
                _press(ser, 'A', duration=1)
                return 0


            while not _color_near(frame[443][1123], (73, 61, 219)):
                _wait_and_render(vid, .1)
                frame = _getframe(vid)
            print('Ready for battle!')

            ended_battle = False
            move_index = 0
            current_text = ''

            while not ended_battle:
                # Beginning of battle
                _press(ser, 'A', sleep_time=1)
                frame = _getframe(vid)
                best_move_index = get_best_move(frame=frame)

                if (best_move_index == -1):
                    print('Uh oh all moves are empty!')
                    return 0

                move_distance = best_move_index - move_index
                move_index = best_move_index

                print(f'Going to use move {best_move_index}')

                _press(ser, 's' if move_distance > 0 else 'w', count=abs(move_distance), sleep_time=0.2)

                _press(ser, 'A', sleep_time=1)

                attacking = True

                while attacking:
                    
                    if (_color_near(frame[443][1123], (73, 61, 219))):
                        attacking = False

                    if _color_near(frame[y_val][x_val], (255, 255, 255)):
                        current_text = get_text(frame=frame, top_left=Point(y=584, x=622), bottom_right=Point(y=641, x=802), invert=True)
                
                    if (current_text == 'Exp. Points!'):
                        ended_battle = True
                        attacking = False
                        print('Battle ended!')
                        end = time.time() + 5
                        while time.time() < end:
                            _press(ser, 'B', sleep_time=0.5)
                    time.sleep(1.75)
                    frame = _getframe(vid) 

    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
