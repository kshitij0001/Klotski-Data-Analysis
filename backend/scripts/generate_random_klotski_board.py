# backend/scripts/generate_random_klotski_board.py


import sys
import os
import random

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.klotski.utils import Block
from backend.scripts.board_generator import render_klotski_board  
from rich.console import Console
from rich.table import Table

console = Console()

def generate_random_klotski_board(rows=5, cols=4):
    """
    Generate a random valid Klotski board configuration.
    - One 2x2 block (goal block), placed randomly (not at (3,1)).
    - Random mix of 2x1, 1x2, and 1x1 blocks.
    - Exactly two empty spaces.
    - No overlaps, all blocks within bounds.
    """
    # Block count tracker
    block_counts = {"2x2": 0, "2x1": 0, "1x2": 0, "1x1": 0}


    # 1. Place the 2x2 block randomly, excluding (3,1)
    possible_2x2_positions = [
        (r, c)
        for r in range(rows - 1)
        for c in range(cols - 1)
        if not (r == 3 and c == 1)  # Exclude the winning position
    ]
    goal_row, goal_col = random.choice(possible_2x2_positions)
    blocks = [Block(2, 2, goal_row, goal_col)]
    block_counts["2x2"] +=1 

    # 2. Mark occupied cells
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for i in range(goal_row, goal_row + 2):
        for j in range(goal_col, goal_col + 2):
            grid[i][j] = '2x2'

    # 3. Fill the rest with random blocks (2x1, 1x2, 1x1) until exactly 2 empty spaces remain
    block_types = [(2, 1), (1, 2), (1, 1)]
    while True:
        empty_cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] is None]
        if len(empty_cells) == 2:
            break  # Stop when exactly 2 empty spaces remain

        random.shuffle(empty_cells)
        placed = False
        for num_rows, num_cols in random.sample(block_types, len(block_types)):
            for (i, j) in empty_cells:
                if i + num_rows <= rows and j + num_cols <= cols:
                    fits = True
                    for di in range(num_rows):
                        for dj in range(num_cols):
                            if grid[i + di][j + dj] is not None:
                                fits = False
                    if fits:
                        blocks.append(Block(num_rows, num_cols, i, j))
                        block_type_str = f"{num_rows}x{num_cols}"
                        block_counts[block_type_str] += 1 
                        for di in range(num_rows):
                            for dj in range(num_cols):
                                grid[i + di][j + dj] = f"{num_rows}x{num_cols}"
                        placed = True
                        break
            if placed:
                break
        # Safety check: only place 1x1 if we have more than 2 empty spaces
        if not placed and len(empty_cells) > 2:
            for (i, j) in empty_cells:
                if grid[i][j] is None:
                    blocks.append(Block(1, 1, i, j))
                    grid[i][j] = "1x1"
                    block_counts["1x1"] +=1
                    break

    return blocks, block_counts



if __name__ == "__main__":
    # Generate Board
    blocks, block_counts = generate_random_klotski_board()
    console.print(f"\n[bold Green]Generated random Klotski board:[/bold Green]")
    render_klotski_board(blocks)

    #Display block counts
    console.print("\n[bold magenta]Block counts used in the board:[/bold magenta]")
    for block_type in ["2x2", "2x1", "1x2", "1x1"]:
        count = block_counts.get(block_type, 0)
        console.print(f" - [bold]{block_type}[/bold]: {count}")