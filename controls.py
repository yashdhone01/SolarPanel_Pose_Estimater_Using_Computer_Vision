import cv2

# Windows specific scancodes via waitKeyEx
KEY_UP_WIN = 2490368
KEY_DOWN_WIN = 2621440
KEY_LEFT_WIN = 2424832
KEY_RIGHT_WIN = 2555904

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
    elif k == ord(' '):
        return 'space'
    elif k == ord('r') or k == ord('R'):
        return 'r'
    elif k == ord('q') or k == ord('Q') or k == 27: # 27 is Esc
        return 'q'
        
    return None
