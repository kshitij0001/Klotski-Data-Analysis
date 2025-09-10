# backend/dashboard/app_html.py

import sys
import os
import copy
import time

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st

from backend.klotski_board import klotski_board as render_board
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board
from backend.klotski.solver import Solver

ROWS, COLS = 5, 4
WIN_ROW, WIN_COL = 3, 1  # 2x2 top-left target

st.set_page_config(page_title="Interactive Klotski Puzzle", layout="centered")
st.title("🧩 Interactive Klotski Puzzle")

def new_solvable(max_tries=120, max_seconds=8.0):
    """Generate a new solvable puzzle"""
    start = time.time()
    for t in range(max_tries):
        blocks, counts = generate_random_klotski_board()
        try:
            s = Solver(blocks, WIN_ROW, WIN_COL)
            s.solve()
            if s.solution_node is not None:
                return blocks, counts, len(s.get_solution_boards()) - 1
        except Exception:
            pass
        if (time.time() - start) > max_seconds:
            break
    return None, None, None

def serialize_blocks(blocks):
    """Convert blocks to format expected by JavaScript component"""
    out = []
    for i, b in enumerate(blocks):
        out.append({
            "id": i,
            "num_rows": b.num_rows,
            "num_cols": b.num_cols,
            "row_pos": b.row_pos,
            "col_pos": b.col_pos,
            "is_goal": (b.num_rows == 2 and b.num_cols == 2),
        })
    return out

# Initialize game state
if "game_initialized" not in st.session_state:
    with st.spinner("Generating initial puzzle..."):
        blocks, counts, min_steps = new_solvable()
        if blocks is None:
            st.error("Could not generate a solvable puzzle. Please refresh the page.")
            st.stop()
        
        st.session_state.game_initialized = True
        st.session_state.initial_blocks = copy.deepcopy(blocks)
        st.session_state.min_steps = min_steps
        st.session_state.player_moves = 0
        st.session_state.game_won = False
        st.session_state.solution_data = None
        st.session_state.solution_step = 0
        st.session_state.viewing_solution = False
        st.session_state.auto_playing = False

# Control buttons
col1, col2, col3 = st.columns(3)

# Handle button clicks directly without passing commands to component
with col1:
    if st.button("🔀 New Puzzle"):
        # Generate a new puzzle immediately
        with st.spinner("Generating new puzzle..."):
            blocks, counts, min_steps = new_solvable()
            if blocks is not None:
                st.session_state.initial_blocks = copy.deepcopy(blocks)
                st.session_state.min_steps = min_steps
                st.session_state.player_moves = 0
                st.session_state.game_won = False
                st.session_state.solution_data = None
                st.session_state.solution_step = 0
                st.session_state.viewing_solution = False
                st.session_state.auto_playing = False
                st.success(f"🎲 New puzzle generated! Minimum steps: {min_steps}")
                st.rerun()
            else:
                st.error("Could not generate a new puzzle. Please try again.")

with col2:
    if st.button("🔄 Reset"):
        # Reset to the initial puzzle state
        st.session_state.player_moves = 0
        st.session_state.game_won = False
        st.session_state.solution_data = None
        st.session_state.solution_step = 0
        st.session_state.viewing_solution = False
        st.session_state.auto_playing = False
        # Force component to reset by updating the key
        if 'reset_counter' not in st.session_state:
            st.session_state.reset_counter = 0
        st.session_state.reset_counter += 1
        st.success("🔄 Puzzle reset!")
        st.rerun()

with col3:
    if st.button("🧠 Get Solution"):
        # Solve the current puzzle immediately
        with st.spinner("Solving puzzle..."):
            try:
                # Use the current blocks
                current_blocks = st.session_state.initial_blocks if hasattr(st.session_state, 'initial_blocks') else []
                if current_blocks:
                    solver = Solver(current_blocks, WIN_ROW, WIN_COL)
                    solver.solve()
                    solution = solver.get_solution_boards()
                    
                    if solution and len(solution) > 1:
                        # Convert solution to the format expected by the component
                        solution_data = []
                        for state in solution:
                            step_data = []
                            for block in state:
                                step_data.append({
                                    'rows': block.num_rows,
                                    'cols': block.num_cols,
                                    'row': block.row_pos,
                                    'col': block.col_pos
                                })
                            solution_data.append(step_data)
                        
                        st.session_state.solution_data = solution_data
                        st.session_state.solution_step = 0
                        st.session_state.viewing_solution = True
                        st.session_state.auto_playing = False
                        st.success(f"✨ Solution found in {len(solution) - 1} steps!")
                        st.rerun()
                    else:
                        st.error("No solution found for this puzzle.")
                else:
                    st.error("No puzzle loaded to solve.")
            except Exception as e:
                st.error(f"Error solving puzzle: {str(e)}")

# Game status display
if hasattr(st.session_state, 'min_steps') and st.session_state.min_steps:
    st.metric("Minimum Steps", st.session_state.min_steps)

# Determine which blocks to display (current state or solution step)
display_blocks = st.session_state.initial_blocks if hasattr(st.session_state, 'initial_blocks') else []
if hasattr(st.session_state, 'viewing_solution') and st.session_state.viewing_solution and st.session_state.solution_data:
    if st.session_state.solution_step < len(st.session_state.solution_data):
        # Convert solution step back to our block format
        solution_step = st.session_state.solution_data[st.session_state.solution_step]
        display_blocks = []
        for i, block_data in enumerate(solution_step):
            from backend.klotski.utils import Block
            block = Block(
                num_rows=block_data['rows'],
                num_cols=block_data['cols'],
                row_pos=block_data['row'],
                col_pos=block_data['col']
            )
            display_blocks.append(block)

# Render the interactive game component
reset_key = f"interactive_klotski_{st.session_state.get('reset_counter', 0)}"
value = render_board(
    blocks=serialize_blocks(display_blocks),
    rows=ROWS,
    cols=COLS,
    minSteps=st.session_state.min_steps if hasattr(st.session_state, 'min_steps') else None,
    command=None,  # No longer passing commands to component
    key=reset_key,
    default=None,
)

# Handle events from the JavaScript component
if value and isinstance(value, dict):
    event_type = value.get("type")
    
    if event_type == "gameUpdate":
        # Update game state from JavaScript (no rerun needed)
        st.session_state.player_moves = value.get("playerMoves", 0)
        st.session_state.game_won = value.get("gameWon", False)
    
    elif event_type == "win":
        st.session_state.game_won = True
        st.session_state.player_moves = value.get("playerMoves", 0)
        min_steps = value.get("minSteps", 0)
        
        # Show win message with performance info
        st.balloons()
        if st.session_state.player_moves == min_steps:
            st.success(f"🎉 Perfect! You solved it in {st.session_state.player_moves} steps - the optimal solution!")
        else:
            st.success(f"🎉 Congratulations! You solved the puzzle in {st.session_state.player_moves} steps! (Optimal: {min_steps})")
        
        # Clear any solution viewing state
        st.session_state.viewing_solution = False
        st.session_state.auto_playing = False
    
    elif event_type == "solution":
        # Store solution and enter solution viewing mode
        solution = value.get("solution", [])
        steps = value.get("steps", 0)
        
        if solution and len(solution) > 0:
            st.session_state.solution_data = solution
            st.session_state.solution_step = 0
            st.session_state.viewing_solution = True
            st.session_state.auto_playing = False
            st.success(f"✨ Solution found in {steps} steps!")
            st.rerun()
        else:
            st.error("No solution found for this puzzle.")
    
    elif event_type == "error":
        st.error(value.get("message", "An error occurred"))

# Instructions
with st.expander("How to Play", expanded=False):
    st.write("""
    **Objective:** Move the large red block (2×2) to the bottom center position.
    
    **How to play:**
    1. Click on any block to select it (it will be highlighted).
    2. Click on a green highlighted area to move the selected block there.
    3. Move the red (2x2) block to bottom center to win.
    4. Try to solve the puzzle in the minimum number of moves!
    
    **Block Colors:**
    - 🟥 **Red (2×2)**: The goal block that needs to reach the bottom center
    - 🟨 **Yellow (1×2)**: Horizontal rectangular blocks  
    - 🟦 **Blue (2×1)**: Vertical rectangular blocks
    - 🟩 **Green (1×1)**: Small square blocks
    """)

# Solution navigation controls
if hasattr(st.session_state, 'viewing_solution') and st.session_state.viewing_solution and st.session_state.solution_data and len(st.session_state.solution_data) > 1:
    st.subheader("Solution Navigation")
    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([1, 1, 1, 2])
    
    with nav_c1:
        if st.button("⬅️ Previous", disabled=(st.session_state.solution_step <= 0 or st.session_state.auto_playing)):
            st.session_state.solution_step = max(0, st.session_state.solution_step - 1)
            st.rerun()
    
    with nav_c2:
        if st.button("Next ➡️", disabled=(st.session_state.solution_step >= len(st.session_state.solution_data)-1 or st.session_state.auto_playing)):
            st.session_state.solution_step = min(len(st.session_state.solution_data)-1, st.session_state.solution_step + 1)
            st.rerun()
    
    with nav_c3:
        # Play/Pause button
        if st.session_state.auto_playing:
            if st.button("⏸️ Pause"):
                st.session_state.auto_playing = False
                st.rerun()
        else:
            play_disabled = st.session_state.solution_step >= len(st.session_state.solution_data) - 1
            if st.button("▶️ Play", disabled=play_disabled):
                st.session_state.auto_playing = True
                st.session_state.last_auto_step_time = time.time()
                st.rerun()
        
    with nav_c4:
        if st.session_state.auto_playing:
            st.write(f"🎬 Auto-playing... Step {st.session_state.solution_step} of {len(st.session_state.solution_data)-1}")
        else:
            st.write(f"Step {st.session_state.solution_step} of {len(st.session_state.solution_data)-1}")
    
    # Exit solution view button
    if st.button("🔙 Back to Puzzle"):
        st.session_state.viewing_solution = False
        st.session_state.auto_playing = False
        st.rerun()

# Auto-play logic - non-blocking and responsive
if hasattr(st.session_state, 'viewing_solution') and st.session_state.viewing_solution and st.session_state.auto_playing:
    # Initialize the auto-play start time
    if 'auto_play_start_time' not in st.session_state:
        st.session_state.auto_play_start_time = time.time()
        st.session_state.auto_play_last_step_time = time.time()
    
    current_time = time.time()
    
    # Advance every 0.8 seconds for smooth playback
    if current_time - st.session_state.auto_play_last_step_time >= 0.8:
        if st.session_state.solution_step < len(st.session_state.solution_data) - 1:
            st.session_state.solution_step += 1
            st.session_state.auto_play_last_step_time = current_time
        else:
            # End reached, stop auto-play
            st.session_state.auto_playing = False
            if 'auto_play_start_time' in st.session_state:
                del st.session_state.auto_play_start_time
                del st.session_state.auto_play_last_step_time
    
    # Auto-refresh for smooth playback only when actively playing
    if st.session_state.auto_playing:
        st.rerun()

# Footer with game stats
if hasattr(st.session_state, 'viewing_solution') and st.session_state.viewing_solution:
    if st.session_state.solution_step == len(st.session_state.solution_data) - 1:
        st.success("🎉 Optimal Solution Complete!")
    else:
        st.info(f"Viewing solution step {st.session_state.solution_step} of {len(st.session_state.solution_data)-1}")
elif st.session_state.game_won:
    st.markdown("---")
    st.markdown("### 🎯 Puzzle Solved!")
    
    # Show performance summary
    if hasattr(st.session_state, 'min_steps') and st.session_state.min_steps:
        moves = st.session_state.player_moves
        optimal = st.session_state.min_steps
        if moves == optimal:
            st.success(f"🌟 **Perfect Score!** You solved it in {moves} steps (optimal solution)")
        else:
            efficiency = round((optimal / moves) * 100, 1)
            st.info(f"📊 You solved it in {moves} steps (optimal: {optimal}) - {efficiency}% efficiency")
    
    st.markdown("**What would you like to do next?**")
    
    # Action buttons in columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 New Puzzle", use_container_width=True):
            # Generate a completely new puzzle
            from backend.scripts.generate_random_klotski_board import generate_random_board
            new_blocks = generate_random_board()
            st.session_state.initial_blocks = new_blocks
            
            # Reset all game state
            st.session_state.player_moves = 0
            st.session_state.game_won = False
            st.session_state.solution_data = None
            st.session_state.solution_step = 0
            st.session_state.viewing_solution = False
            st.session_state.auto_playing = False
            st.session_state.min_steps = None
            
            st.success("🎲 New puzzle generated! Good luck!")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset This Puzzle", use_container_width=True):
            # Reset current puzzle to its starting state
            st.session_state.player_moves = 0
            st.session_state.game_won = False
            st.session_state.solution_data = None
            st.session_state.solution_step = 0
            st.session_state.viewing_solution = False
            st.session_state.auto_playing = False
            # Force component to reset by updating the key
            if 'reset_counter' not in st.session_state:
                st.session_state.reset_counter = 0
            st.session_state.reset_counter += 1
            st.success("🔄 Puzzle reset!")
            st.rerun()
            
            st.success("🔄 Puzzle reset! Try to beat your previous score!")
            st.rerun()
