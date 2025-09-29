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

st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-image: url("https://img.freepik.com/premium-photo/empty-wooden-table-top-with-dark-background-product-presentation_1304147-2943.jpg?semt=ais_hybrid&w=740&q=80");
        background-attachment: fixed;
        background-size: cover;
    }

    /* Import medieval font */
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');

    /* Golden medieval text */
    h1, h2, h3, h4, h5, h6, p, span, .stMarkdown, .stText, .stButton button {
        font-family: 'UnifrakturCook', cursive !important;
        color: #FFD700 !important; /* gold */
        text-shadow: 1px 1px 3px #000000; /* gives depth */
    }

    /* Board frame for camera */
    .board-frame {
        border: 10px solid #5a3e1b;
        border-radius: 15px;
        padding: 5px;
        background: rgba(210,180,140,0.8);
        box-shadow: 0 0 25px rgba(0,0,0,0.6);
    }

    div.stButton > button:first-child {
        font-family: 'UnifrakturCook', cursive !important;
        font-size: 28px !important;
        font-weight: bold !important;
        color: #FFD700 !important; /* Gold text */
        text-shadow: 2px 2px 4px #000000; /* Depth */
        background: linear-gradient(145deg, #5a3e1b, #3b2a15); /* Medieval wood/bronze feel */
        border: 4px solid #FFD700;
        border-radius: 12px;
        padding: 12px 40px;
        box-shadow: 0px 0px 15px rgba(0,0,0,0.7);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(145deg, #3b2a15, #5a3e1b);
        transform: scale(1.05);
        box-shadow: 0px 0px 25px rgba(255,215,0,0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
game, rules = st.tabs(["Game","Rules"])

with rules:
    rules.image("./images/rules.jpg")

with game:
    game.text_input("Game name", key = "game_name", value = "temp")
    game.selectbox("Player Number", ["One", "Two"], key = "player_num")
    st.session_state_opponent_num = ["One", "Two"].remove(st.session_state.player_num)
    if f'{st.session_state.game_name}.pkl' in os.listdir("./games/"):
        with open(f'./games/{st.session_state.game_name}.pkl', 'rb') as file:
            st.session_state.scores = pickle.load(file)
    else:
        st.session_state.scores = {
            "One": 0,
            "Two": 0
        }

    game.header(f"Your Score: {st.session_state.scores[st.session_state.player_num]}")

    game.checkbox("Enable Capture", key="start_capture")
    cap = cv2.VideoCapture(1)  # Capture the video
    if not cap.isOpened():
        game.error("Could not open webcam or video source.")

    rolled, selected = game.columns([1, 1])
    with rolled:
        frame_placeholder = st.empty()
        while cap.isOpened() and st.session_state.start_capture:    
            success, frame = cap.read()
            if not success:
                game.warning("Failed to read frame from webcam. Please verify the webcam is connected properly.")
                break

            results = st.session_state.model.track(
                frame, conf=0.4, iou=0.5, persist=True
            )

            st.session_state.selections =  results[0].boxes.cls.cpu().numpy().astype(int)
            annotated_frame = results[0].plot()  # Add annotations on frame

            frame_placeholder.image(annotated_frame, channels="BGR", caption="Predicted Frame")  # Display processed
            # time.sleep(0.5)

        cap.release() 
        cv2.destroyAllWindows()
        
    with selected:
        
        if len(st.session_state.selections) > 0:
            selected.header("Dice Counts")
            selected.write(st.session_state.selections + 1) 
        dist = [0, 0, 0, 0, 0, 0]
        for val in st.session_state.selections:
            dist[val] += 1
        max_score = calculate_score(game_state = dist)
        selected.header("Selected: " + str(max_score))
        
    if game.button("Next Roll") and max_score > 0:
        st.session_state.current_score += max_score
        st.session_state.selections = []
        st.rerun()
    # st.write(st.session_state)
    game.header(f"Your Turn Socre is: {st.session_state.current_score}")
    if st.button("⚔️ Pass"):
        with open(f'./games/{st.session_state.game_name}.pkl', 'wb') as file:
            st.session_state.scores[st.session_state.player_num] += st.session_state.current_score
            pickle.dump(st.session_state.scores, file, protocol=pickle.HIGHEST_PROTOCOL)
            st.session_state.selections = []
            st.session_state.current_score = 0
        st.rerun()