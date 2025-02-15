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

class Color(NamedTuple):
    b: int
    g: int
    r: int

def _press(ser: serial.Serial, s: str, duration: float = .1) -> None:
    # print(f'{s=} {duration=}')
    ser.write(s.encode())
    time.sleep(duration)
    ser.write(b'0')
    time.sleep(.075)

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

def increment_counter(frame: numpy.ndarray, delay: float, file_path="counter.txt", ):
    counter_path = Path(file_path)
    log_path = Path("log.txt")
    
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
    log_data = f'Count: {count} - Delay: {delay} - Timestamp {timestamp}'

    # Write the updated count back to the file
    with counter_path.open("w") as file1, log_path.open("a") as file2:
        file1.write(str(count))
        file2.write(log_data + '\n')
    
    cv2.imwrite(f"/Volumes/Untitled/poke screenshots/{count}.png", frame)
    

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-BG009FDF')
    args = parser.parse_args()

    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  
    with serial.Serial(args.serial, 9600) as ser, _shh(ser):
        while True:
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

            _press(ser, 'w', duration=.5)

            # Get to where we are looking at turtwig
            while not _color_near(frame[510][1000], (255, 255, 255)):
                _press(ser, 'A')
                time.sleep(.20)
                frame = _getframe(vid)

            _wait_and_render(vid, 1)
            
            # Select yes
            _press(ser, 'w', duration=.25)
            _press(ser, 'A', duration=.25)

            _wait_and_render(vid, 1.5)

            x_val = 960
            y_val = 660

            _await_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('started battle!') 
            _wait_and_render(vid, 1)
            _await_not_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255),exact_pixel=False)
            print('starting pixel gone')
            _wait_and_render(vid, 4)

            while not _color_near(frame[y_val][x_val], (255, 255, 255)):
                print(frame[y_val][x_val])
                _wait_and_render(vid, .1)
                frame = _getframe(vid)
           
            _wait_and_render(vid, .25)
            print('You encountered a starly!')
            _await_not_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('starly pixel gone') 
            
            _await_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))
            print('Go Turtwig!')
            _await_not_pixel(ser, vid, x=x_val, y=y_val, pixel=(255, 255, 255))

            print('dialog ended')
            t0 = time.time()

            _await_pixel(ser, vid, x=35, y=630, pixel=(250, 250, 250), exact_pixel=False)

            t1 = time.time()
            delay = t1 - t0
            print(f'dialog delay: {delay:.3f}s')
            frame = _getframe(vid)

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
                increment_counter(frame=frame, delay=delay)
                return 0

            increment_counter(frame=frame, delay=delay)
    


    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
