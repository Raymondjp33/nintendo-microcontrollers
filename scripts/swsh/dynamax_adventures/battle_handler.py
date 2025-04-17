import time
from typing import NamedTuple
from utils import Point, _press, _getframe, get_text, _color_near
import cv2
import numpy
import serial
from config_manager import ConfigManager

# E for effectivness and P for PP
class Move(NamedTuple):
    e: str
    p: int
    index: int

class BattleHandler:
    def __init__(self, vid: cv2.VideoCapture, ser: serial.Serial, config: ConfigManager):
        self.vid = vid
        self.ser = ser
        self.config = config

    def handle_fight(self):
        print('Fighting')
        _press(self.ser, 'A', sleep_time=1.5)

        self.dynamax_if_available()
        self.attack_with_move()

    def attack_with_move(self):
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

        frame = _getframe(self.vid)
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
        
        dynamax_turns = self.config.get('dynamax_turns')
        move_index = self.config.get('move_index')
        if (dynamax_turns is not None):
            dynamax_turns -= 1

        if (dynamax_turns is not None and dynamax_turns < 0):
            dynamax_turns = None
            move_index = 0

        print(f'move1: {move1E} {move1P}\nmove2: {move2E} {move2P}\nmove3: {move3E} {move3P}\nmove4: {move4E} {move4P}')
        new_move_index = sorted_data[0].index
        print(f'About to use move: {new_move_index}\nCurrently at move {move_index}')
        distance = move_index - new_move_index

        _press(self.ser, 's' if distance < 0 else 'w', count=abs(distance), sleep_time=0.2)
        _press(self.ser, 'A', sleep_time=1.5)
        _press(self.ser, 'A', sleep_time=0.75)
        
        # Attempt using move on self if using it in general failed
        _press(self.ser, 's', sleep_time=0.5)
        _press(self.ser, 'A', sleep_time=0.5)

        # Go back if all else fails
        _press(self.ser, 'B', sleep_time=0.5, count=3)
        self.config.update({'dynamax_turns': dynamax_turns, 'move_index':new_move_index})
    
    def dynamax_if_available(self):
        frame = _getframe(self.vid)

        dynamax_available = False

        for _ in range(10):
            if (_color_near(frame[590][728], (79, 0, 222))):
                dynamax_available = True
            time.sleep(0.1)
            frame = _getframe(self.vid)
        # print(f'Here! {frame[590][728]}')
        if (dynamax_available):
            print('Dynamaxing!')
            self.config.update({'dynamax_turns': 3})
            _press(self.ser, 'a', sleep_time=0.5)
            _press(self.ser, 'A', sleep_time=0.5)
        else:
            print('Not dynamaxing!')