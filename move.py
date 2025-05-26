# move_simple.py
import sys
import tty
import termios
from base_ctrl import BaseController

# 제어 기본값
SPEED = 0.10
ROTATE_SPEED = 0.35

base = BaseController('/dev/ttyUSB0', 115200)

def send_control(L, R):
    base.base_json_ctrl({"T": 1, "L": L, "R": R})
    print(f"🚗 L: {L:.2f}, R: {R:.2f}")

def stop():
    send_control(0.0, 0.0)

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

print("""
🕹️ 제어 키:
  w - 앞으로
  s - 뒤로ww
  a - 왼쪽 회전
  d - 오른쪽 회전
  space - 정지
  q - 종료
""")

try:
    while True:
        key = getch()

        if key == 'w':
            send_control(SPEED, SPEED)
        elif key == 's':
            send_control(-SPEED, -SPEED)
        elif key == 'a':
            send_control(-ROTATE_SPEED, ROTATE_SPEED)
        elif key == 'd':
            send_control(ROTATE_SPEED, -ROTATE_SPEED)
        elif key == ' ':
            stop()
        elif key == 'q':
            break
        else:
            print("❓ 알 수 없는 키:", key)

except KeyboardInterrupt:
    pass

finally:
    print("🛑 정지 중...")
    stop()
