import json
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigHandler(FileSystemEventHandler):
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def on_modified(self, event):
        if event.src_path == self.config_manager.file_path:
            self.config_manager.reload()

class ConfigManager:
    def __init__(self):
        self.file_path = '/Users/raymondprice/Desktop/other/test_coding/pokemon_scripts/nintendo-microcontrollers/scripts/swsh/dynamax_adventures/den_config.json'
        self._lock = threading.Lock()
        self._data = self._load()

        event_handler = ConfigHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, path='.', recursive=False)
        self.observer.start()

    def _load(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def reload(self):
        with self._lock:
            # print(f"[Watcher] Reloading config")
            self._data = self._load()

    def get(self, key, default=None):
        self.reload()
        with self._lock:
            return self._data.get(key, default)

    def update(self, updates: dict):
        self.reload()
        with self._lock:
            self._data.update(updates)
            with open(self.file_path, 'w') as f:
                json.dump(self._data, f, indent=4)

    def stop(self):
        self.observer.stop()
        self.observer.join()