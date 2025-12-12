# digit_app_final_corrected_v3_save_screens.py
import pygame, sys
from pygame.locals import *
import numpy as np
import cv2
from keras.models import load_model
import os
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# CONFIG
# -----------------------------
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)

MODEL_PATH = "bestmodel_png.h5"
IMAGESAVE = True
PREDICT = True

LABELS = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
}

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    MODEL = load_model(MODEL_PATH)
    print("Model Loaded")
except Exception as e:
    print("Could not load model:", e)
    sys.exit()

pygame.init()
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition Project")

font_big = pygame.font.Font("freesansbold.ttf", 36)
font_med = pygame.font.Font("freesansbold.ttf", 24)
font_small = pygame.font.Font("freesansbold.ttf", 18)

os.makedirs("saved_images", exist_ok=True)

# track which screen is currently shown (1..5)
current_screen = 0

def save_current_screen():
    """Save current displayed screen to screen{n}.png (overwrite mode)."""
    global current_screen
    if current_screen == 0:
        # unknown screen - default to screen1
        fname = "screen1.png"
    else:
        fname = f"screen{current_screen}.png"
    try:
        pygame.image.save(DISPLAYSURF, fname)
        print(f"Saved current screen as: {fname}")
    except Exception as e:
        print("Failed to save screen:", e)

# =========================================================
# CENTER + PREPARE FOR CNN
# =========================================================
def center_and_prepare(roi):
    if roi is None or roi.size == 0:
        return None
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = roi.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    resized = cv2.resize(roi, (new_w, new_h))
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    canvas = canvas.astype("float32") / 255.0
    return np.expand_dims(canvas, axis=(0, -1))


# =========================================================
# PREDICT A CROPPED DIGIT
# =========================================================
def predict_from_crop(binary_img, x, y, w, h):
    roi = binary_img[y:y+h, x:x+w]
    prep = center_and_prepare(roi)
    if prep is None:
        return None, 0.0
    pred = MODEL.predict(prep, verbose=0)[0]
    return int(np.argmax(pred)), float(np.max(pred))


# =========================================================
# DETECT DIGITS IN UPLOADED IMAGE
# =========================================================
def detect_digits_in_image_cv(path):
    img = cv2.imread(path)
    if img is None:
        return "", None, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [(x, y, w, h) for x, y, w, h in (cv2.boundingRect(c) for c in contours) if w * h > 80]
    boxes = sorted(boxes)

    ann = img.copy()
    results = []
    digits = ""

    for (x, y, w, h) in boxes:
        d, c = predict_from_crop(thresh, x, y, w, h)
        if d is None:
            continue
        digits += str(d)
        results.append((x, y, w, h, d, c))
        cv2.rectangle(ann, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(ann, f"{d} ({c*100:.1f}%)", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    return digits, ann, results


# =========================================================
# DRAW MODE (Unchanged)
# =========================================================
def run_draw_mode():
    global current_screen
    current_screen = 3  # screen 3 = blackboard/draw
    DISPLAYSURF.fill(BLACK)
    pygame.display.update()

    iswriting = False
    xs, ys = [], []
    image_cnt = 1

    while True:
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit(); sys.exit()

            if e.type == MOUSEMOTION and iswriting:
                x, y = e.pos
                pygame.draw.circle(DISPLAYSURF, WHITE, (x, y), 6, 0)
                xs.append(x); ys.append(y)

            if e.type == MOUSEBUTTONDOWN:
                iswriting = True

            if e.type == MOUSEBUTTONUP:
                iswriting = False
                if xs and ys:
                    x1, x2 = max(min(xs)-5,0), min(max(xs)+5,WINDOWSIZEX)
                    y1, y2 = max(min(ys)-5,0), min(max(ys)+5,WINDOWSIZEY)
                    xs, ys = [], []

                    arr3 = pygame.surfarray.array3d(DISPLAYSURF)
                    region = arr3[x1:x2, y1:y2]
                    gray = cv2.cvtColor(np.transpose(region,(1,0,2)), cv2.COLOR_BGR2GRAY)

                    img = cv2.resize(gray,(28,28))
                    img = np.pad(img,(10,10),constant_values=0)
                    img = cv2.resize(img,(28,28))/255.0

                    pred = MODEL.predict(img.reshape(1,28,28,1), verbose=0)
                    d = int(np.argmax(pred))
                    c = float(np.max(pred))*100

                    lbl = font_small.render(f"{d} ({c:.1f}%)", True, ORANGE)
                    rect = lbl.get_rect()
                    rect.left, rect.bottom = x1, y2

                    pygame.draw.rect(DISPLAYSURF, RED,(x1,y1,x2-x1,y2-y1),2)
                    DISPLAYSURF.blit(lbl, rect)

            if e.type == KEYDOWN:
                # Save current screen
                if e.unicode == "s" or e.key == K_s:
                    save_current_screen()
                if e.key == K_BACKSPACE:
                    return
                if e.unicode == "c":
                    DISPLAYSURF.fill(BLACK)
                if e.unicode == "q":
                    pygame.quit(); sys.exit()

        pygame.display.update()


# =========================================================
# FILE SELECT DIALOG
# =========================================================
def choose_files_dialog():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        filetypes=[("Image Files","*.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return list(files)


# =========================================================
# SCREEN 5
# =========================================================
def show_upload_message_screen():
    global current_screen
    current_screen = 5  # screen 5 = post-upload message
    # clean screen and show box with message
    DISPLAYSURF.fill(BLACK)
    # draw a centered rounded-ish box (rectangle)
    box_w, box_h = 520, 200
    box_x = (WINDOWSIZEX - box_w) // 2
    box_y = (WINDOWSIZEY - box_h) // 2
    # rectangle background
    pygame.draw.rect(DISPLAYSURF, (40,40,40), (box_x, box_y, box_w, box_h))
    # border
    pygame.draw.rect(DISPLAYSURF, GRAY, (box_x, box_y, box_w, box_h), 2)

    lines = [
        "Press ENTER to upload more images",
        "Press BACKSPACE to go back",
        "Press Q to quit"
    ]
    # render lines centered
    y = box_y + 20
    for i, ln in enumerate(lines):
        if i == 0:
            surf = font_med.render(ln, True, WHITE)
        else:
            surf = font_small.render(ln, True, (200,200,200))
        DISPLAYSURF.blit(surf, (box_x + (box_w - surf.get_width()) // 2, y))
        y += surf.get_height() + 8

    pygame.display.update()

    # wait for user input
    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); sys.exit()
            if ev.type == KEYDOWN:
                # Save current screen
                if ev.unicode == "s" or ev.key == K_s:
                    save_current_screen()
                if ev.key == K_RETURN:
                    return "enter"
                if ev.key == K_BACKSPACE:
                    return "back"
                if ev.unicode == "q":
                    pygame.quit(); sys.exit()
        pygame.time.delay(30)


# =========================================================
# UPLOAD MODE (modified to implement 4->5 flow)
# =========================================================
def run_upload_mode():
    global current_screen
    while True:
        files = choose_files_dialog()
        if not files:
            return  # user cancelled -> back to mode selection

        for path in files:
            # Screen 4: show annotated detection immediately
            digits, ann, results = detect_digits_in_image_cv(path)
            if ann is None:
                # couldn't read image -> skip
                continue

            # prepare and display annotated image (centered)
            h, w = ann.shape[:2]
            scale = min(WINDOWSIZEX / w, WINDOWSIZEY / h, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            ann_res = cv2.resize(ann, (new_w, new_h), interpolation=cv2.INTER_AREA)
            ann_rgb = cv2.cvtColor(ann_res, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(np.transpose(ann_rgb, (1, 0, 2)))

            current_screen = 4  # screen 4 = uploaded image detection
            DISPLAYSURF.fill(BLACK)
            DISPLAYSURF.blit(surf, ((WINDOWSIZEX - new_w) // 2, (WINDOWSIZEY - new_h) // 2))

            # display prediction + confidence (use the same confidence shown on the boxes)
            if results:
                conf = results[0][5] * 100
                txt = f"Predicted: {digits}    Confidence: {conf:.1f}%"
            else:
                txt = "No digits detected"

            txtS = font_med.render(txt, True, RED)
            DISPLAYSURF.blit(txtS, (10, WINDOWSIZEY - 70))

            # instructions for next step on same screen (tell user to press ENTER to proceed)
            instr = font_small.render("Press ENTER-message screen | BACKSPACE-back | Q-quit", True, GREEN)
            DISPLAYSURF.blit(instr, (10, WINDOWSIZEY - 40))
            pygame.display.update()

            # wait on Screen 4 for ENTER / BACKSPACE / Q / S
            while True:
                for ev in pygame.event.get():
                    if ev.type == QUIT:
                        pygame.quit(); sys.exit()
                    if ev.type == KEYDOWN:
                        # Save current screen
                        if ev.unicode == "s" or ev.key == K_s:
                            save_current_screen()
                        if ev.key == K_RETURN:
                            # go to Screen 5
                            next_action = show_upload_message_screen()
                            if next_action == "enter":
                                # user chose to upload more images: break to outer loop to re-open file dialog
                                break  # exit this inner while -> continue handling outer for-loop/files
                            elif next_action == "back":
                                return  # go back to mode selection
                        if ev.key == K_BACKSPACE:
                            return  # go back to mode selection immediately
                        if ev.unicode == "q":
                            pygame.quit(); sys.exit()
                else:
                    # executed if no break -> continue waiting
                    pygame.time.delay(30)
                    continue
                # if we reach here, a break happened (user pressed ENTER and then ENTER on message)
                break

            # save annotated image for record
            try:
                save_name = f"saved_images/uploaded_{os.path.basename(path)}"
                cv2.imwrite(save_name, ann)
            except Exception:
                pass

        # after finishing this batch of files, loop to allow selecting more files immediately
        # (behavior: after pressing ENTER on message screen, user lands here and will be prompted again)

    # end run_upload_mode


# =========================================================
# MODE SELECTION
# =========================================================
def run_mode_selection():
    global current_screen
    while True:
        current_screen = 2  # screen 2 = mode selection
        DISPLAYSURF.fill(BLACK)
        t = font_big.render("Choose Mode", True, WHITE)
        o1 = font_med.render("1 - Draw on Blackboard", True, GRAY)
        o2 = font_med.render("2 - Upload Image", True, GRAY)
        hint = font_small.render("1 or 2 | Backspace to return | Q quit | S save", True, GRAY)

        DISPLAYSURF.blit(t, ((WINDOWSIZEX - t.get_width()) // 2, 130))
        DISPLAYSURF.blit(o1, ((WINDOWSIZEX - o1.get_width()) // 2, 210))
        DISPLAYSURF.blit(o2, ((WINDOWSIZEX - o2.get_width()) // 2, 260))
        DISPLAYSURF.blit(hint, ((WINDOWSIZEX - hint.get_width()) // 2, WINDOWSIZEY - 50))

        pygame.display.update()

        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit(); sys.exit()
            if e.type == KEYDOWN:
                # Save current screen
                if e.unicode == "s" or e.key == K_s:
                    save_current_screen()
                if e.key == K_1:
                    run_draw_mode()
                if e.key == K_2:
                    run_upload_mode()
                if e.key == K_BACKSPACE:
                    return
                if e.unicode == "q":
                    pygame.quit(); sys.exit()
        pygame.time.delay(30)


# =========================================================
# START SCREEN
# =========================================================
def run_start_screen():
    global current_screen
    current_screen = 1  # screen 1 = start
    DISPLAYSURF.fill(BLACK)
    t = font_big.render("Digit Recognition", True, WHITE)
    h = font_small.render("Press ENTER to continue", True, GREEN)
    s = font_small.render("Shortcuts: C clear | S save | Q quit | BACKSPACE back", True, GRAY)

    DISPLAYSURF.blit(t, ((WINDOWSIZEX - t.get_width()) // 2, 150))
    DISPLAYSURF.blit(h, ((WINDOWSIZEX - h.get_width()) // 2, 225))
    DISPLAYSURF.blit(s, ((WINDOWSIZEX - s.get_width()) // 2, WINDOWSIZEY - 40))
    pygame.display.update()

    while True:
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit(); sys.exit()
            if e.type == KEYDOWN:
                # Save current screen
                if e.unicode == "s" or e.key == K_s:
                    save_current_screen()
                if e.key == K_RETURN:
                    return
                if e.unicode == "q":
                    pygame.quit(); sys.exit()
        pygame.time.delay(30)


# =========================================================
# MAIN
# =========================================================
def main():
    while True:
        run_start_screen()
        run_mode_selection()

if __name__ == "__main__":
    main()
