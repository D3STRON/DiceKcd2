import streamlit as st
from collections import defaultdict
from itertools import chain, combinations
import numpy as np
import os
import random
import pandas as pd
import pickle

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


if "selections" not in st.session_state:
    st.session_state.selections = defaultdict(int)
if "rolls" not in st.session_state:
    st.session_state.rolls = []
if "current_score" not in st.session_state:
    st.session_state.current_score = 0
if "dices" not in st.session_state:
    st.session_state.dices = 6

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
# st.header()
# Roll button
if st.button("Roll"):
    st.session_state.rolls = [random.choice([1,2,3,4,5,6]) for _ in range(6)]
st.write("🎲 Rolled:", st.session_state.rolls)


enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)

rolled, selected = st.columns([1, 1])

with rolled:
    col1, col2 = st.columns(2)

    def render_section(num, label):
        st.header(label)
        for i, val in enumerate(st.session_state.rolls):
            if val == num:
                key = f"{num}_{i}"
                checked = st.checkbox(f"{val}", key=f"chk_{key}")




    # Left column (odds)
    with col1:
        render_section(1, "Ones")
        render_section(3, "Threes")
        render_section(5, "Fives")

    # Right column (evens)
    with col2:
        render_section(2, "Twos")
        render_section(4, "Fours")
        render_section(6, "Sixes")

st.session_state.selections = defaultdict(int)
for key in st.session_state:
    if key.startswith("chk_") and st.session_state[key]:
        value = key.split("_")[1]
        st.session_state.selections[int(value)] += 1

    
with selected:
    st.header("Counts")
    st.write(st.session_state.selections)
    
    max_score = calculate_score(game_state = np.array([ st.session_state.selections[face] for face in range(1,7)]))
    st.write(max_score)
    
if st.button("Roll Again") and max_score > 0:
    st.session_state.current_score += max_score
    st.session_state.dices -= sum(st.session_state.selections.values())
    st.session_state.rolls = [random.choice([1,2,3,4,5,6]) for _ in range(st.session_state.dices)]
    for key in st.session_state:
        if key.startswith("chk_"):
            del st.session_state[key]
    
    st.session_state.selections = defaultdict(int)
    st.rerun()
# st.write(st.session_state)
st.header(f"You Current Socre is: {st.session_state.current_score}")

if st.button("Pass"):
    with open(f'./games/{st.session_state.game_name}.pkl', 'wb') as file:
        st.session_state.scores[st.session_state.player_num] += st.session_state.current_score
        pickle.dump(st.session_state.scores, file, protocol=pickle.HIGHEST_PROTOCOL)
        st.session_state.dices = 6
        st.session_state.rolls = []
        st.session_state.selections = defaultdict(int)
        st.session_state.current_score = 0
    st.rerun()