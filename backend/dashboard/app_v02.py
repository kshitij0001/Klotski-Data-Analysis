# backend/dashboard/app.py

import sys
import os
import time
import copy

import streamlit as st

# Add backend to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.klotski_board import klotski_board as render_board
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
from backend.klotski.solver import Solver
from backend.klotski.utils import Block

ROWS, COLS = 5, 4
WIN_ROW, WIN_COL = 3, 1  # Standard winning position

st.set_page_config(page_title="Klotski Solver", layout="centered")
st.title("🧩 Klotski Solver")

# --- Helper functions ---

def serialize_blocks(blocks):
    out = []
    for i, b in enumerate(blocks):
        out.append({
            "id": i,
            "numRows": b.num_rows,
            "numCols": b.num_cols,
            "rowPos": b.row_pos,
            "colPos": b.col_pos,
            "is_goal": (b.num_rows == 2 and b.num_cols == 2),  # optional
        })
    return out

def deserialize_blocks(block_dicts):
    """Convert dicts to Block objects."""
    return [
        Block(
            num_rows=b["num_rows"],
            num_cols=b["num_cols"],
            row_pos=b["row_pos"],
            col_pos=b["col_pos"]
        ) for b in block_dicts
    ]

def show_instructions():
    st.markdown(
        """
        <div style="background:#f8f9fa;padding:1em;border-radius:8px;border:1px solid #ddd;">
        <b>How to Play:</b>
        <ol>
        <li>Click <b>New Puzzle</b> to generate a random, solvable Klotski board.</li>
        <li>Click <b>Solve</b> to find the optimal solution.</li>
        <li>Use <b>Next</b> and <b>Prev</b> to step through the solution, or <b>Auto Play</b> to animate.</li>
        <li>The goal is to move the <span style="color:#d9534f;font-weight:bold;">red 2x2 block</span> to the pink target area.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True
    )

def board_color_legend():
    st.markdown(
        """
        <div style="margin-bottom:1em;">
        <span style="display:inline-block;width:1.5em;height:1.5em;background:#d9534f;border-radius:4px;margin-right:0.5em;"></span> 2x2 (Goal Block)
        <span style="display:inline-block;width:1.5em;height:1.5em;background:#0275d8;border-radius:4px;margin-left:1em;margin-right:0.5em;"></span> 2x1 (Vertical)
        <span style="display:inline-block;width:1.5em;height:1.5em;background:#f0ad4e;border-radius:4px;margin-left:1em;margin-right:0.5em;"></span> 1x2 (Horizontal)
        <span style="display:inline-block;width:1.5em;height:1.5em;background:#5cb85c;border-radius:4px;margin-left:1em;margin-right:0.5em;"></span> 1x1 (Small)
        </div>
        """, unsafe_allow_html=True
    )

# --- Streamlit State Initialization ---

if "blocks" not in st.session_state:
    blocks, _ = generate_random_klotski_board()
    st.session_state.blocks = blocks
    st.session_state.solution = []
    st.session_state.step = 0
    st.session_state.solved = False
    st.session_state.loading = False

# --- UI Layout ---

show_instructions()
board_color_legend()

# --- Controls ---

col1, col2, col3, col4 = st.columns([1,1,1,2])
with col1:
    if st.button("🔀 New Puzzle"):
        blocks, _ = generate_random_klotski_board()
        st.session_state.blocks = blocks
        st.session_state.solution = []
        st.session_state.step = 0
        st.session_state.solved = False
        st.session_state.loading = False
with col2:
    if st.button("✅ Reset"):
        st.session_state.blocks = copy.deepcopy(st.session_state.blocks)
        st.session_state.solution = []
        st.session_state.step = 0
        st.session_state.solved = False
        st.session_state.loading = False
with col3:
    if st.button("🧠 Solve"):
        st.session_state.loading = True
        solver = Solver(st.session_state.blocks, WIN_ROW, WIN_COL)
        solver.solve()
        solution = solver.get_solution_boards()
        st.session_state.solution = solution
        st.session_state.step = 0
        st.session_state.solved = bool(solution)
        st.session_state.loading = False

# --- Board Display ---

st.subheader("Current Board")
blocks_to_show = (
    st.session_state.solution[st.session_state.step]
    if st.session_state.solution else st.session_state.blocks
)
render_board(
    blocks=serialize_blocks(blocks_to_show),
    rows=ROWS,
    cols=COLS,
    winningRow=WIN_ROW,
    winningCol=WIN_COL,
    key="klotski_board",
)

# --- Solution Controls ---

if st.session_state.solution:
    st.success(f"Solution found in {len(st.session_state.solution)-1} moves!")
    colA, colB, colC, colD = st.columns([1,1,1,2])
    with colA:
        if st.button("⏮️ Prev", disabled=st.session_state.step == 0):
            st.session_state.step = max(0, st.session_state.step - 1)
    with colB:
        if st.button("⏭️ Next", disabled=st.session_state.step == len(st.session_state.solution)-1):
            st.session_state.step = min(len(st.session_state.solution)-1, st.session_state.step + 1)
    with colC:
        if st.button("▶️ Auto Play"):
            for i in range(st.session_state.step, len(st.session_state.solution)):
                st.session_state.step = i
                time.sleep(0.3)
                st.experimental_rerun()
    with colD:
        st.markdown(f"<div style='margin-top:0.5em;'>Step: <b>{st.session_state.step}</b> / <b>{len(st.session_state.solution)-1}</b></div>", unsafe_allow_html=True)
else:
    st.info("Click 'Solve' to find the optimal solution.")

st.markdown("---")
st.caption("© Klotski Solver — Streamlit Edition")
