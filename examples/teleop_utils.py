import json
import os
import threading

import eventlet
import socketio
from pynput import keyboard


class KeyboardClient:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.pressed_keys.add(key)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key)
        except AttributeError:
            pass


class IPhoneClient:
    def __init__(self, port: int = 5555, silent: bool = True):
        self._port = port
        self._silent = silent
        self._sio = socketio.Server(cors_allowed_origins="*")
        self._app = socketio.WSGIApp(self._sio)
        self._latest_data = {}
        self._commands = list()

        # Set up the event handler for updates
        @self._sio.event
        def update(sid, data):
            self._latest_data = json.loads(data)
            self._sio.emit("commands", json.dumps(self._commands), to=sid)

    def _run_server(self):
        eventlet.wsgi.server(
            eventlet.listen(("", self._port)), self._app, log=open(os.devnull, "w") if self._silent else None
        )

    def start(self):
        threading.Thread(target=self._run_server, daemon=True).start()

    def get_latest_data(self):
        return self._latest_data

    def set_haptic_feedback(self, enable: bool):
        self._commands = ["start_haptics" if enable else "stop_haptics"]
