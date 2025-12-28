import threading
import queue
import pyautogui
from collections import deque

class ActionWorker(threading.Thread):
    """Exécute les actions PyAutoGUI dans un thread séparé"""
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.5)
                if item is None:
                    # Sentinel de stop
                    self.queue.task_done()
                    break

                action_type, args = item

                if action_type == "move":
                    pyautogui.moveTo(*args)
                elif action_type == "click":
                    pyautogui.click()
                elif action_type == "scroll":
                    pyautogui.scroll(*args)
                elif action_type == "key":
                    pyautogui.press(*args)
                elif action_type == "hotkey":
                    pyautogui.hotkey(*args)
                elif action_type == "custom":
                    func, f_args = args
                    func(*f_args)

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WORKER ERROR] {e}")

    def add_action(self, action_type, args=None):
        with self.queue.mutex:
            if action_type == "move":
                self.queue.queue = deque([item for item in self.queue.queue if item and item[0] != "move"])
            elif action_type == "scroll":
                # Option robuste anti-inertie : purge tous les scroll en attente
                self.queue.queue = deque([item for item in self.queue.queue if item and item[0] != "scroll"])

        self.queue.put((action_type, args))

    def stop(self):
        # Stop immédiat
        self.running = False
        try:
            self.queue.put(None)  # Sentinel
        except Exception:
            pass
        self.join(timeout=2.0)
