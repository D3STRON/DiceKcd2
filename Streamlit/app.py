import streamlit as st
from collections import defaultdict
import numpy as np
import os
import random
import pickle
import cv2
from ultralytics import YOLO
import time

def calculate_score(game_state):
    game_state = np.array(game_state, dtype=int)
    max_score = 0
    
    if np.all(game_state > 0):
        new_state = game_state - 1
        score = 1500 + calculate_score(tuple(new_state))
        max_score = max(max_score, score)

    if np.all(game_state[1:] > 0):
        new_state = game_state.copy()
        new_state[1:] -= 1
        score = 750 + calculate_score(tuple(new_state))
        max_score = max(max_score, score)

    if np.all(game_state[:-1] > 0):
        new_state = game_state.copy()
        new_state[:-1] -= 1
        score = 500 + calculate_score(tuple(new_state))
        max_score = max(max_score, score)

    score = 0
    for i, count in enumerate(game_state):
        if count >= 3:
            base = 1000 if i == 0 else (i + 1) * 100
            score += base * (1 << (count - 3)) 
            game_state[i] = 0
        elif i == 0:
            score += 100 * count
            game_state[i] = 0
        elif i == 4:
            score += 50 * count
            game_state[i] = 0
    if np.sum(game_state) == 0:
        max_score = max(max_score, score)
    return max_score

model_loc = model_name = "../models/cls.pt"

if "selections" not in st.session_state:
    st.session_state.selections = []
if "rolls" not in st.session_state:
    st.session_state.rolls = []
if "current_score" not in st.session_state:
    st.session_state.current_score = 0
if "model" not in st.session_state:
    st.session_state.model = YOLO(model_name)



st.title("Dice!")
st.text_input("Game name", key = "game_name", value = "temp")
st.selectbox("Player Number", ["One", "Two"], key = "player_num")
st.session_state_opponent_num = ["One", "Two"].remove(st.session_state.player_num)
if f'{st.session_state.game_name}.pkl' in os.listdir("./games/"):
    with open(f'./games/{st.session_state.game_name}.pkl', 'rb') as file:
        st.session_state.scores = pickle.load(file)
else:
    st.session_state.scores = {
        "One": 0,
        "Two": 0
    }

st.header(f"Your Score: {st.session_state.scores[st.session_state.player_num]}")

st.checkbox("Enable Capture", key = "start_capture")
cap = cv2.VideoCapture(1)  # Capture the video
if not cap.isOpened():
    st.error("Could not open webcam or video source.")

rolled, selected = st.columns([1, 1])
with rolled:
    frame_placeholder = st.empty()
    while cap.isOpened() and st.session_state.start_capture:    
        success, frame = cap.read()
        if not success:
            st.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
            break

        results = st.session_state.model.track(
            frame, conf=0.4, iou=0.5, persist=True
        )

        st.session_state.selections =  results[0].boxes.cls.cpu().numpy().astype(int)
        annotated_frame = results[0].plot()  # Add annotations on frame

        frame_placeholder.image(annotated_frame, channels="BGR", caption="Predicted Frame")  # Display processed
        time.sleep(0.5)

    cap.release() 
    cv2.destroyAllWindows()
    
with selected:
    
    if len(st.session_state.selections) > 0:
        st.header("Dice Counts")
        st.write(st.session_state.selections + 1) 
    dist = [0, 0, 0, 0, 0, 0]
    for val in st.session_state.selections:
        dist[val] += 1
    max_score = calculate_score(game_state = dist)
    st.header("Selected: " + str(max_score))
    
if st.button("Next Roll") and max_score > 0:
    st.session_state.current_score += max_score
    st.session_state.selections = []
    st.rerun()
# st.write(st.session_state)
st.header(f"Your Turn Socre is: {st.session_state.current_score}")

if st.button("Pass"):
    with open(f'./games/{st.session_state.game_name}.pkl', 'wb') as file:
        st.session_state.scores[st.session_state.player_num] += st.session_state.current_score
        pickle.dump(st.session_state.scores, file, protocol=pickle.HIGHEST_PROTOCOL)
        st.session_state.selections = []
        st.session_state.current_score = 0
    st.rerun()