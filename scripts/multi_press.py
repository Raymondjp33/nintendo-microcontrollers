from __future__ import annotations

import argparse
import time

import serial

def main() -> int:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--serial', default=SERIAL_DEFAULT)
    parser.add_argument('--serial', default='/dev/tty.usbserial-120') # Switch 2
    # parser.add_argument('--serial', default='/dev/tty.usbmodem1101') # Switch 1
    
    parser.add_argument('--duration', type=float, default=.1)
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()

    with serial.Serial(args.serial, 9600) as ser:
        time.sleep(1)
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == 'exit':
                break

            elif user_input:  # Filter single character keys
                ser.write(user_input.encode())
                time.sleep(args.duration)
                ser.write(b'.')
                time.sleep(.05)
                print(f"Sent: {user_input}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
