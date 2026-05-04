import cv2

# Windows specific scancodes via waitKeyEx
KEY_UP_WIN = 2490368
KEY_DOWN_WIN = 2621440
KEY_LEFT_WIN = 2424832
KEY_RIGHT_WIN = 2555904

mouse_click_pos = None

def mouse_callback(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < 640 and y < 480:
            mouse_click_pos = (x, y)

def setup_mouse(window_name):
    cv2.setMouseCallback(window_name, mouse_callback)

def get_mouse_click():
    global mouse_click_pos
    pos = mouse_click_pos
    mouse_click_pos = None
    return pos

def process_input(delay=1):
    """
    Waits for key press for `delay` ms.
    Returns string mapping of key pressed, or None.
    """
    k = cv2.waitKeyEx(delay)
    if k == -1:
        return None
        
    if k == KEY_UP_WIN or k == 82 or k == 63232:  # Windows, Linux, Mac
        return 'up'
    elif k == KEY_DOWN_WIN or k == 84 or k == 63233:
        return 'down'
    elif k == KEY_LEFT_WIN or k == 81 or k == 63234:
        return 'left'
    elif k == KEY_RIGHT_WIN or k == 83 or k == 63235:
        return 'right'
    elif k == ord('w') or k == ord('W'):
        return 'w'
    elif k == ord('s') or k == ord('S'):
        return 's'
    elif k == ord('a') or k == ord('A'):
        return 'a'
    elif k == ord('d') or k == ord('D'):
        return 'd'
    elif k == ord(' '):
        return 'space'
    elif k == ord('r') or k == ord('R'):
        return 'r'
    elif k == ord('q') or k == ord('Q') or k == 27: # 27 is Esc
        return 'q'
        
    return None
