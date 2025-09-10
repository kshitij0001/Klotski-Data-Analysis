# backend/dashboard/app.py

import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# backend/dashboard/app.py

import streamlit as st
from backend.klotski.solver import Solver
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
from backend.klotski.utils import Block

import numpy as np
from PIL import Image, ImageDraw


# --- Helper to render board ---
def draw_board(blocks, rows=5, cols=4, cell=80):
    img = Image.new("RGB", (cols*cell, rows*cell), (245,245,245))
    d = ImageDraw.Draw(img)

    # grid
    for y in range(rows):
        for x in range(cols):
            d.rectangle([x*cell, y*cell, (x+1)*cell, (y+1)*cell], outline=(200,200,200))

    # blocks
    for b in blocks:
        x, y = b.col_pos, b.row_pos
        w, h = b.num_cols, b.num_rows
        color = (200,100,100) if (w==2 and h==2) else (100,150,200)
        d.rectangle([x*cell+4, y*cell+4, (x+w)*cell-4, (y+h)*cell-4], fill=color)
    return img


# --- Streamlit UI ---
st.set_page_config(page_title="Klotski Solver Dashboard", layout="centered")
st.title("🧩 Klotski Puzzle Solver")

if "blocks" not in st.session_state:
    st.session_state.blocks, _ = generate_random_klotski_board()

col1, col2 = st.columns(2)

with col1:
    if st.button("🔀 Generate Puzzle"):
        st.session_state.blocks, _ = generate_random_klotski_board()

with col2:
    if st.button("✅ Solve Puzzle"):
        solver = Solver(st.session_state.blocks, 3, 1)
        solver.solve()
        st.session_state.solution = solver.get_solution_boards()

# Display current puzzle
st.subheader("Current Puzzle")
st.image(draw_board(st.session_state.blocks))

# If solved, animate solution
if "solution" in st.session_state:
    if not st.session_state.solution or len(st.session_state.solution) == 0:
        st.error("❌ This puzzle is unsolvable.")
    else:
        st.subheader("Solution Path")
        steps = st.slider("Step", 0, len(st.session_state.solution)-1, 0)
        st.image(draw_board(st.session_state.solution[steps]))
        st.write(f"Step {steps} of {len(st.session_state.solution)-1}")
