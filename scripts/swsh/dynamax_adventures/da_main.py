from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Generator

import cv2
import serial
from battle_handler import BattleHandler
from den_handler import DenHandler
from config_manager import ConfigManager
import utils 


@contextlib.contextmanager
def _shh(ser: serial.Serial) -> Generator[None]:
    try:
        yield
    finally:
        ser.write(b'.')    

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='/dev/tty.usbserial-120')
    args = parser.parse_args()

    vid = cv2.VideoCapture(2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    shiny_legend = False

    config = ConfigManager()

    with serial.Serial(args.serial, 9600) as ser, _shh(ser):
        time.sleep(1)

        battle_handler = BattleHandler(vid, ser, config)
        den_handler = DenHandler(vid, ser, config)
        # go_to_change_grip(ser)
        # connect_and_go_to_game(ser)
        # handle_choose_pokemon()
        # take_pokemon(ser)
        # restart_dungeon(vid, ser)
        # select_starter(vid, ser)
        # handle_choose_pokemon(vid, ser)
        # return 0
        
        while not shiny_legend:

            screen = utils.get_screen(vid)

            if (screen == 'Fight'):
                battle_handler.handle_fight()
            
            if (screen == 'Swapping'):
                den_handler.swap_if_needed()

            if (screen == 'Catching'):
                config.update({'move_index': 0, 'battle_index': config.get('battle_index') + 1,'dynamax_turns': None})
                den_handler.handle_catch(is_legend=config.get('battle_index')==4)

            if (screen == 'Selecting'):
                den_handler.select_starter()

            if (screen == 'Choosing'):
                shiny_legend = den_handler.handle_choose_pokemon()

                if (shiny_legend):
                    break

                den_handler.restart_dungeon(keep_dungeon=config.get('keep_dungeon'))
                # return 0

            if (screen == 'Cheer On'):
                if (config.get('dynamax_turns') is not None):
                    config.update({'dynamax_turns': -1})
                print('Cheering')
                utils._press(ser, 'A')

            if (screen == 'Let down'):
                print('Handling let down')
                den_handler.restart_dungeon(keep_dungeon=config.get('keep_dungeon'))

            if (screen == 'Pathing'):
                den_handler.handle_pathing()

            if (screen == 'Items'):
                den_handler.handle_select_item()

            if (screen == 'Rental'):
                den_handler.handle_rental()

            if (screen == 'Sus'):
                den_handler.handle_sus()

            time.sleep(5)

    vid.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
