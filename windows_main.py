import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import webbrowser
import subprocess
import time
import platform

# Set PyAutoGUI to fail-safe (move mouse to upper-left corner to stop)
pyautogui.FAILSAFE = True

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define the virtual keyboard layout with additional keys
keys = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'BACKSPACE'],
    ['TAB', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['CAPS', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'ENTER'],
    ['SHIFT', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', 'SPACE'],
    ['CTRL', 'ALT', 'COPY', 'PASTE', 'YOUTUBE', 'GOOGLE', 'CURSOR']
]

# Keyboard dimensions
key_width, key_height = 60, 60
key_spacing = 50  # Space between keys
keyboard_x, keyboard_y = 20, 60  # Position of the keyboard on the screen

# Colors (BGR format for OpenCV)
BLUE = (255, 0, 0)    # Blue for default key color
GREEN = (0, 255, 0)   # Green for hover
WHITE = (255, 255, 255)
RED = (0, 0, 255)     # Red for selection indicator

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# State variables
modifier_states = {'SHIFT': False, 'CTRL': False, 'ALT': False, 'CAPS': False}
last_selected_key = None
selecting_text = False
last_hand_x = None
last_hand_y = None
selection_cooldown = 0.15  # Adjusted cooldown for smoother selection
selection_dwell_time = 0.3  # Dwell time to confirm selection start
selection_start_time = 0
pinch_cooldown = 0.5  # Cooldown for pinch gesture to prevent rapid presses
last_pinch_time = 0
pinch_threshold = 0.08  # Increased threshold for pinch detection (was 0.05 in earlier versions)

def draw_keyboard(frame, hover_key=None, is_selecting=False):
    # Draw the keyboard
    for row in range(len(keys)):
        for col in range(len(keys[row])):
            key = keys[row][col]
            x = keyboard_x + col * (key_width + key_spacing)
            y = keyboard_y + row * (key_height + key_spacing)
            
            # Determine key color
            color = GREEN if hover_key == (row, col) else BLUE
            if key in modifier_states and modifier_states[key]:
                color = (0, 255, 255)  # Yellow for active modifiers
            
            # Draw the key
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), color, -1)
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), WHITE, 2)
            
            # Draw the key label
            text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x + (key_width - text_size[0]) // 2
            text_y = y + (key_height + text_size[1]) // 2
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    
    # Add visual feedback for selection state
    status_text = "Selecting: " + ("ON" if is_selecting else "OFF")
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, RED if is_selecting else WHITE, 2)

def get_key_at_position(x, y):
    for row in range(len(keys)):
        for col in range(len(keys[row])):
            key_x = keyboard_x + col * (key_width + key_spacing)
            key_y = keyboard_y + row * (key_height + key_spacing)
            if key_x <= x <= key_x + key_width and key_y <= y <= key_y + key_height:
                return row, col
    return None

def get_pinch_distance(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance

def is_hand_folded(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    fingertips = [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.PINKY_TIP]
    ]
    # Check distance between wrist and fingertips
    distance_threshold = 0.2  # Increased threshold for easier detection
    for tip in fingertips:
        distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        if distance > distance_threshold:
            return False
    
    # Additional check: angle between finger joints to confirm closed hand
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calculate vectors
    vec1 = np.array([index_pip.x - index_mcp.x, index_pip.y - index_mcp.y])
    vec2 = np.array([index_tip.x - index_pip.x, index_tip.y - index_pip.y])
    
    # Calculate angle between vectors
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return False
    cos_angle = dot_product / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
    
    # If the angle is small (finger is bent), consider the hand folded
    return angle > 90  # Adjust this threshold as needed

# Main loop
last_press_time = 0
last_selection_update = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    
    hover_key = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            finger_x = int(index_tip.x * w)
            finger_y = int(index_tip.y * h)
            
            # Check if hand is folded for text selection
            hand_folded = is_hand_folded(hand_landmarks.landmark)
            current_time = time.time()
            
            if hand_folded:
                if not selecting_text:
                    if selection_start_time == 0:
                        selection_start_time = current_time
                    elif current_time - selection_start_time >= selection_dwell_time:
                        selecting_text = True
                        last_hand_x = finger_x
                        last_hand_y = finger_y
                # Move selection in any direction
                if selecting_text and last_hand_x is not None and last_hand_y is not None:
                    if current_time - last_selection_update >= selection_cooldown:
                        # Adjusted sensitivity for smoother selection
                        if finger_x < last_hand_x - 30:  # Moved left
                            pyautogui.hotkey('shift', 'left')
                        elif finger_x > last_hand_x + 30:  # Moved right
                            pyautogui.hotkey('shift', 'right')
                        if finger_y < last_hand_y - 30:  # Moved up
                            pyautogui.hotkey('shift', 'up')
                        elif finger_y > last_hand_y + 30:  # Moved down
                            pyautogui.hotkey('shift', 'down')
                        last_selection_update = current_time
                last_hand_x = finger_x
                last_hand_y = finger_y
            else:
                selecting_text = False  # Stop extending selection, but selection persists in the app
                selection_start_time = 0
                last_hand_x = None
                last_hand_y = None
            
            # Skip key detection if selecting text
            if selecting_text:
                continue
            
            # Check if finger is over a key
            hover_key = get_key_at_position(finger_x, finger_y)
            
            # Pinch gesture for key selection
            pinch_distance = get_pinch_distance(hand_landmarks.landmark)
            if hover_key is not None and pinch_distance < pinch_threshold:
                if current_time - last_pinch_time >= pinch_cooldown:
                    row, col = hover_key
                    key = keys[row][col]
                    # Handle modifier keys
                    if key in ['SHIFT', 'CTRL', 'ALT']:
                        modifier_states[key] = not modifier_states[key]
                    elif key == 'CAPS':
                        modifier_states['CAPS'] = not modifier_states['CAPS']
                    elif key == 'COPY':
                        pyautogui.hotkey('ctrl', 'c')
                    elif key == 'PASTE':
                        pyautogui.hotkey('ctrl', 'v')
                    elif key == 'YOUTUBE':
                        webbrowser.open('https://www.youtube.com')
                    elif key == 'GOOGLE':
                        webbrowser.open('https://www.google.com')
                    elif key == 'CURSOR':
                        if platform.system() == 'Windows':
                            subprocess.run(['cmd', '/c', 'code'])
                        elif platform.system() == 'Darwin':  # macOS
                            subprocess.run(['open', '-a', 'Cursor'])
                        elif platform.system() == 'Linux':
                            subprocess.run(['code'])
                    else:
                        # Handle regular key presses with modifiers
                        if modifier_states['CTRL']:
                            pyautogui.hotkey('ctrl', key.lower())
                        elif modifier_states['ALT']:
                            pyautogui.hotkey('alt', key.lower())
                        elif modifier_states['SHIFT'] or modifier_states['CAPS']:
                            if key == 'SPACE':
                                pyautogui.press('space')
                            elif key == 'ENTER':
                                pyautogui.press('enter')
                            elif key == 'BACKSPACE':
                                pyautogui.press('backspace')
                            elif key == 'TAB':
                                pyautogui.press('tab')
                            elif key == '<':
                                pyautogui.press('left')
                            elif key == '>':
                                pyautogui.press('right')
                            else:
                                pyautogui.write(key)
                        else:
                            if key == 'SPACE':
                                pyautogui.press('space')
                            elif key == 'ENTER':
                                pyautogui.press('enter')
                            elif key == 'BACKSPACE':
                                pyautogui.press('backspace')
                            elif key == 'TAB':
                                pyautogui.press('tab')
                            elif key == '<':
                                pyautogui.press('left')
                            elif key == '>':
                                pyautogui.press('right')
                            else:
                                pyautogui.write(key.lower())
                    last_pinch_time = current_time
            last_selected_key = hover_key
    
    # Draw the keyboard with selection state
    draw_keyboard(frame, hover_key, selecting_text)
    
    # Show the frame
    cv2.imshow("Virtual Keyboard", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()