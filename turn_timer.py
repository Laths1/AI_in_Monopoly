import time
import threading

class Timer90:
    def __init__(self):
        self.time_left = 90
        self.running = False

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._run)
        thread.start()

    def _run(self):
        while self.time_left > 0 and self.running:
            print(f"Time left: {self.time_left} sec", end="\r")
            time.sleep(1)
            self.time_left -= 1
        print("Time Limit! - Next Turn")
        
    def stop(self):
        self.running = False
