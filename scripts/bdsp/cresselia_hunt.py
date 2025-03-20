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

def increment_counter(frame: numpy.ndarray, delay: float, file_path="cress-counter.txt", ):
    counter_path = Path(file_path)
    log_path = Path("cress-log.txt")
    
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
    star = '* ' if delay > 0.53 or delay < 0.47 else ''
    log_data = f'{star}Count: {count} - Delay: {delay} - Timestamp {timestamp}'

    # Write the updated count back to the file
    with counter_path.open("w") as file1, log_path.open("a") as file2:
        file1.write(str(count))
        file2.write(log_data + '\n')
    
    cv2.imwrite(f"/Volumes/Untitled/poke screenshots/{count}.png", frame)

def get_roamer_on_route(ser: serial.Serial, vid: cv2.VideoCapture,):
         
        # for _ in range(7):
        #     frame = _getframe(vid)
        #     print(frame[117][1038])
        #     time.sleep(0.15)
        # return 
    
        in_route = False
        while True: 


            if (not in_route):
                _press(ser, 'd', duration=0.35)
                in_route = True
            
            if (in_route):
                for _ in range(6):
                    frame = _getframe(vid)
                    if (_color_near(frame[117][1038], (31, 63, 28))):
                        double_check = double_check_mes_on_route(vid)
                        if (double_check):
                            print('It is here!')
                            
                            return       
                    # print('Not here!')
                    time.sleep(0.10)
                _press(ser, 'a')
                in_route = False


def double_check_mes_on_route(vid: cv2.VideoCapture,):
        for _ in range(6):
            frame = _getframe(vid)
            if (_color_near(frame[117][1038], (31, 63, 28))):
                # print('It is here!')
                return True
            # print('Not here!')
            time.sleep(0.1)
        
        return False

def write_shiny_text():
    shiny_text_path = Path(f"shiny_text.txt")
    with shiny_text_path.open("w") as file1:
        file1.write('I got the shiny! My switch\nwill be off until I am\nback. Make sure to come\nback when/after I catch it!')


def encounter_mespirit(ser: serial.Serial, vid: cv2.VideoCapture,):
    _press(ser, 'X', sleep_time=1)
    _press(ser, 'd', sleep_time=0.5, count=2)
    _press(ser, 'w', sleep_time=0.5)
    _press(ser, 'A', sleep_time=1.5)
    _press(ser, 'd', count=4)
    _press(ser, 'A', sleep_time=1, count=2)
    _press(ser, 'B', sleep_time=1, count=4)
    _press(ser, 's', duration=.2)
    _press(ser, 'd', duration=1)

    mes_e = False

    while True:
            go_right = False
            frame = _getframe(vid)
            time_check = time.time() + 240
            while not _color_near(frame[300][300], (250, 250, 250)):
                _press(ser, 'd' if go_right else 'a', count=2)
                _wait_and_render(vid, .15)
                frame = _getframe(vid)
                go_right = not go_right
                if (time_check < time.time()):
                    current_text = get_text(frame=frame, top_left=Point(y=587, x=295), bottom_right=Point(y=636, x=503), invert=True)
                    if (current_text == 'Repel’s effect'):
                        print('In time check, using another repel')
                        _press(ser, 'A', sleep_time=1.5)
                        _press(ser, 'A', sleep_time=1.5)
                        _press(ser, 'B', sleep_time=1.5)
                        _press(ser, 'B', sleep_time=1.5)
                        time_check = time.time() + 240

            
            for _ in range(60):
                frame = _getframe(vid)
                current_text = get_text(frame=frame, top_left=Point(y=581, x=126), bottom_right=Point(y=656, x=403), invert=True)
                # print(f'here and current text is ${current_text}')
                if (current_text == 'Cresselia appeared!'):
                    mes_e = True
                    print('Cresselia appeared!')
                    break
                time.sleep(0.1)
            
            if (not mes_e):
                print('Did not encounter mespirit')
                time.sleep(15)
                return -1
            log_frame = _getframe(vid)
            x_val = 960
            y_val = 660
            _await_not_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('Cresselia appeared gone') 
            t0 = time.time()
            
            _await_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('Go Dialga!')
            

            t1 = time.time()
            delay = t1 - t0
            print(f'dialog delay: {delay:.3f}s')

            if (delay) > 0.6:
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
                increment_counter(frame=log_frame, delay=delay)
                write_shiny_text()
                return 1

            increment_counter(frame=log_frame, delay=delay)
            return 0

def interact_with_roamer(ser: serial.Serial):
    _press(ser, 'A', sleep_time=7)
    _press(ser, 'A', sleep_time=2)
    _press(ser, 'A', sleep_time=3)

def get_to_route(ser: serial.Serial):
    _press(ser, 's', duration=2 ,sleep_time=3)
    _press(ser, 'X', sleep_time=1)
    _press(ser, 'A', sleep_time=1.5)
    _press(ser, '1', count=8)
    _press(ser, '2', count=8)
    _press(ser, 'A', sleep_time=1.5)
    _press(ser, 'A', sleep_time=9)

    # Should be in front of pokemon center here
    _press(ser, 'a', duration=1)
    _press(ser, 'w', duration=1)
    _press(ser, 'd', duration=4)
    _press(ser, 's', duration=1.05)
    _press(ser, 'd', duration=3.65)
    _press(ser, 'w', duration=0.5)
    _press(ser, 'R')

def reset_game(ser: serial.Serial, vid: cv2.VideoCapture,):
    _press(ser, 'H')
    _wait_and_render(vid, 1)
    _press(ser, 'X')
    _wait_and_render(vid, 1)
    _press(ser, 'A')

    frame = _getframe(vid)
    while not _color_near(frame[50][90], (250, 250, 250)):
        _press(ser, 'A')
        _wait_and_render(vid, .15)
        frame = _getframe(vid)

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
        
        # reset_game(ser=ser, vid=vid,) 
        # interact_with_roamer(ser=ser)
        # get_to_route(ser=ser)
        # get_roamer_on_route(ser=ser, vid=vid,)
        # print(encounter_mespirit(ser=ser, vid=vid))
        # return 0
    
        mes_is_shiny = False
        while not mes_is_shiny:
            start_time = time.time()
            reset_game(ser=ser, vid=vid,) 
            interact_with_roamer(ser=ser)
            get_to_route(ser=ser)

            get_roamer_on_route(ser=ser, vid=vid,)
            encounter_res = encounter_mespirit(ser=ser, vid=vid)

            if (encounter_res == 1):
                mes_is_shiny = True
            
            end_time=time.time()
            print(f'Run took {end_time-start_time:.3f}s')
            # return 0
                


    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
