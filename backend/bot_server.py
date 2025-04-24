from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import os, json, time
from threading import Thread
import eventlet

eventlet.monkey_patch()

app = Flask(__name__, static_folder='/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/stream-browser/stream_browser/build/web')
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS

switch1_file_path = 'switch1_data.json'
switch2_file_path = 'switch2_data.json'
stream_data_file_path = 'stream_data.json'

modified_vars = {
    'switch1_last_modified' : 0,
    'switch2_last_modified' : 0,
    'stream_data_last_modified' : 0
    }


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

def watch_files():
    global last_modified, file_data
    while True:
        handle_file_update(switch1_file_path, 'switch1_last_modified', 'switch1_data')
        handle_file_update(switch2_file_path, 'switch2_last_modified', 'switch2_data')
        handle_file_update(stream_data_file_path, 'stream_data_last_modified', 'stream_data')
       
        time.sleep(2)

def handle_file_update(file_path, modified_var, emit_variable):
    try:
        if not os.path.exists(file_path):
            return
        mtime = os.path.getmtime(file_path)
        if mtime == modified_vars[modified_var]:
            return

        file_data = {}
        with open(file_path) as f:
            file_data = json.load(f)
        modified_vars[modified_var] = mtime
        print(f"[WebSocket] Emitting update: {file_data}")
        socketio.emit(emit_variable, file_data)
    except Exception as e:
            print(f"Error reading file: {e}")

# http://0.0.0.0:5050
if __name__ == '__main__':
    socketio.start_background_task(target=watch_files)
    socketio.run(app, host='0.0.0.0', port=5050)

