import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
import torch
import ctypes
import threading
from pynput import keyboard
import time

model = YOLO('best.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

PUL = ctypes.POINTER(ctypes.c_ulong)

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("mi", MouseInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


MOUSE_LEFTDOWN = 0x0002
MOUSE_LEFTUP = 0x0004
MOUSE_MOVE = 0x0001
MOUSE_ABSOLUTE = 0x8000

# screenkk_width = ctypes.windll.user32.GetSystemMetrics(0)
# screen_height = ctypes.windll.user32.GetSystemMetrics(1)

# 4K
screen_width = 3840
screen_height = 2160


def move_mouse_smooth(x, y, duration=0.1, steps=200):
    while True:
        current_position = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(current_position))
        start_x = current_position.x
        start_y = current_position.y

        delta_x = (x - start_x) / steps
        delta_y = (y - start_y) / steps
        delay = duration / steps

        for i in range(steps):
            new_x = int(start_x + delta_x * (i + 1))
            new_y = int(start_y + delta_y * (i + 1))
            fx = new_x * 65535 // screen_width
            fy = new_y * 65535 // screen_height

            input_struct = Input(ctypes.c_ulong(0), Input_I(
                mi=MouseInput(fx, fy, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, ctypes.pointer(ctypes.c_ulong(0)))))
            ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))

        # time.sleep(delay)

        ctypes.windll.user32.GetCursorPos(ctypes.byref(current_position))
        final_x = current_position.x
        final_y = current_position.y

        if abs(final_x - x) < 2 and abs(final_y - y) < 5:
            print(f"final position: ({final_x}, {final_y})，target position: ({x}, {y})")
            click_mouse_left()
            break

def click_mouse_left():
    input_struct = Input(ctypes.c_ulong(0),
                         Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFTDOWN, 0, ctypes.pointer(ctypes.c_ulong(0)))))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))

    # time.sleep(0.1)

    input_struct = Input(ctypes.c_ulong(0),
                         Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFTUP, 0, ctypes.pointer(ctypes.c_ulong(0)))))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))


def move_mouse_and_click(x, y):
    fx = x * 65535 // screen_width
    fy = y * 65535 // screen_height

    input_struct = Input(ctypes.c_ulong(0), Input_I(
        mi=MouseInput(fx, fy, 0, MOUSE_MOVE | MOUSE_ABSOLUTE, 0, ctypes.pointer(ctypes.c_ulong(0)))))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))

    input_struct = Input(ctypes.c_ulong(0),
                         Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFTDOWN, 0, ctypes.pointer(ctypes.c_ulong(0)))))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))

    input_struct = Input(ctypes.c_ulong(0),
                         Input_I(mi=MouseInput(0, 0, 0, MOUSE_LEFTUP, 0, ctypes.pointer(ctypes.c_ulong(0)))))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(input_struct))

is_detecting = False

def detect_humans():
    global is_detecting

    sct = mss()
    monitor = {"top": 760, "left": 1600, "width": 640, "height": 640}
    while is_detecting:

        time.sleep(0.1)

        screenshot = np.array(sct.grab(monitor))
        img_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        results = model(img_rgb)

        best_bbox = None
        best_confidence = 0
        best_center_x = 0
        best_center_y = 0
        best_label = ""

        for result in results:
            for bbox, confidence, cls_idx in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = [int(i) for i in bbox]
                label = result.names[int(cls_idx)]

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bbox = (x1, y1, x2, y2)
                    best_center_x = center_x
                    best_center_y = center_y
                    best_label = label

        if best_bbox is not None and best_confidence > 0.8:
            x1, y1, x2, y2 = best_bbox

            print(
                f"Highest confidence {best_label} detected at center: [{best_center_x}, {best_center_y}] with confidence {best_confidence:.2f}")

            screen_center_x = best_center_x + monitor["left"]
            screen_center_y = best_center_y + monitor["top"]

            move_mouse_smooth(screen_center_x, screen_center_y)

            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{best_label} {best_confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLO Screen Detection", img_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_detecting = False
            break

    cv2.destroyAllWindows()

def on_press(key):
    global is_detecting
    try:
        if key.char == 'k':
            if not is_detecting:
                print("start detecting")
                is_detecting = True
                detection_thread = threading.Thread(target=detect_humans)
                detection_thread.start()
            else:
                print("stop detecting")
                is_detecting = False
    except AttributeError:
        pass

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()