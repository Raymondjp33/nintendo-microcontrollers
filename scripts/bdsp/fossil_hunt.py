from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time
from collections.abc import Generator
from typing import NamedTuple

import cv2
import numpy
import serial
import functools

from typing import Protocol
import tesserocr

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

def increment_counter(frame: numpy.ndarray,current_color, file_path="fossil-counter.txt", ):
    counter_path = Path(file_path)
    log_path = Path("fossil-log.txt")
    
    # Read the existing count (default to 0 if file does not exist)
    if counter_path.exists():
        with counter_path.open("r") as file:
            try:
                count = int(file.read().strip())
            except ValueError:
                count = 0
    else:
        count = 0

    # Increment the counter
    count += 1

    timestamp  = time.strftime('%Y-%m-%d %H:%M:%S')
    star = '*' if current_color[0] < 222 or current_color[0] > 231 or current_color[1] < 175 or current_color[1] > 182 or current_color[2] < 114 or current_color[2] > 121 else ''
    log_data = f'{star}Count: {count} - Current Color: {current_color} - Timestamp {timestamp}'


    # Write the updated count back to the file
    with counter_path.open("w") as file1, log_path.open("a") as file2:
        file1.write(str(count))
        file2.write(log_data + '\n')
    
    cv2.imwrite(f"/Volumes/Untitled/poke screenshots/{count}.png", frame)

def get_outside(ser: serial.Serial):
    print('Going outside')
    _press(ser, 's', duration=0.5)
    _press(ser, 'a')
    _press(ser, 's', duration=1)
    time.sleep(3)

def get_to_counter(ser: serial.Serial):
    print('Going back to counter')
    _press(ser, 'w', duration=0.5)
    time.sleep(4)
    _press(ser, 'w', duration=1)
    _press(ser, 'd')
    _press(ser, 'w', duration=0.5)

def give_fossil(ser: serial.Serial):
    print('Giving fossil')
    _press(ser, 'A', sleep_time=1, count=3)
    ## Press yes to give fossil
    _press(ser, 'A', sleep_time=1)
    _press(ser, 's', count=3)
    _press(ser, 'A', sleep_time=1, count=3)

def get_pokemon(ser: serial.Serial):
    print('Getting pokemon')
    _press(ser, 'A', sleep_time=1, count=5)

def reset_game(ser: serial.Serial, vid: cv2.VideoCapture,):
    _press(ser, 'H')
    _wait_and_render(vid, 1)
    _press(ser, 'X')
    _wait_and_render(vid, 1)
    _press(ser, 'A')

    for _ in range(26):
        _press(ser, 'A')
        time.sleep(1)

    print('game loaded!')

def connect_and_go_to_game(ser: serial.Serial):
    _press(ser, 'H')
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

class PokemonInfo(NamedTuple):
    base_color: Color
    point: Point
    count: int

    

poke_colors = {
    'omanyte': PokemonInfo(count = 1, point = Point(439, 601), base_color = (250, 228, 121)),
    'lileep': PokemonInfo(count = 1, point = Point(464, 668), base_color = (217, 163, 175)),
    'anorith': PokemonInfo(count = 1, point = Point(446, 645), base_color = (176, 210, 175)),
    'cranidos':  PokemonInfo(count = 3, point = Point(251, 573), base_color = (225, 177, 115)),
}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-BG009FDF')
    args = parser.parse_args()

    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  
    with serial.Serial(args.serial, 9600) as ser, _shh(ser):

        # go_to_change_grip(ser)
        # connect_and_go_to_game(ser)
        # return 0

        while True:
            t0 = time.time()
            reset_game(ser=ser, vid=vid)
            time.sleep(7)
            # for name, info in poke_colors.items():
            info = poke_colors['cranidos']
            for _ in range(info.count):  # Repeat based on count
                give_fossil(ser=ser)
                get_outside(ser=ser)
                get_to_counter(ser=ser)
                get_pokemon(ser=ser)
                time.sleep(3)
                frame = _getframe(vid)
                current_color = frame[info.point.y][info.point.x]
                if (not _color_near(current_color, info.base_color)):
                    print('Shiny!!!')
                    return 0
                print(f'Not Shiny')
                increment_counter(frame=frame, current_color=current_color)
                _press(ser, 'A', sleep_time=2)
            t1 = time.time()
            print(f'Run took {t1-t0:.3f}s')
    
        return 0
    
                


    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
