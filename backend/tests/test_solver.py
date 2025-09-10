# backend/tests/test_solver.py
import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.klotski.utils import Block
from backend.klotski.solver import Solver
from rich.console import Console
from rich.table import Table
from time import sleep

console = Console()

def block_type_symbol(block):
    """Return a symbol based on block type."""
    if block.num_rows == 2 and block.num_cols == 2:
        return "[bold red][ ][/bold red]"  # 2x2 (goal)
    elif block.num_rows == 2 and block.num_cols == 1:
        return "[cyan][ ][/cyan]"  # 2x1 vertical
    elif block.num_rows == 1 and block.num_cols == 2:
        return "[yellow][ ][/yellow]"  # 1x2 horizontal
    elif block.num_rows == 1 and block.num_cols == 1:
        return "[green][ ][/green]"  # 1x1 small
    else:
        return "[black][ ][/black]"  # Unknown/invalid

def render_board(blocks):
    grid = [["  " for _ in range(4)] for _ in range(5)]
    highlight_positions = [(3, 1), (3, 2), (4, 1), (4, 2)]

    # Place block symbols on the grid
    for block in blocks:
        symbol = block_type_symbol(block)
        for i in range(block.row_pos, block.row_pos + block.num_rows):
            for j in range(block.col_pos, block.col_pos + block.num_cols):
                grid[i][j] = symbol

    # Highlight the specific positions regardless of block presence
    for (i, j) in highlight_positions:
        # Wrap current symbol (which could be empty "  ") in highlight markup
        grid[i][j] = f"[on #301c2d]{grid[i][j]}[/on #301c2d]"

    table = Table(show_header=False, box=None)
    for row in grid:
        table.add_row(*row)
    console.print(table)

if __name__ == "__main__":
    # Define the default Klotski puzzle configuration.
    # Each Block is defined as Block(num_rows, num_cols, row_pos, col_pos)
    # The board is 5 rows x 4 columns.
    # this is a board setup with 32 move solution

    blocks = [
        Block(2,2,0,1),     # 2x2 block
        Block(1,1,0,0),     # 1st 1x1
        Block(1,1,0,3),     # 2nd 1x1
        Block(1,1,1,0),     # 3rd 1x1
        Block(1,1,1,3),     # 4th 1x1
        Block(1,2,2,1),     # 1st 1x2 
        Block(2,1,3,0),     # 1st 2x1
        Block(2,1,3,1),     # 2nd 2x1
        Block(2,1,3,2),     # 3rd 2x1
        Block(2,1,3,3),     # 4th 2x1
    ]
    winning_row, winning_col = 3, 1

    solver = Solver(blocks, winning_row, winning_col)
    solver.solve()
    solution = solver.get_solution_boards()

    if solution:
        console.print(f"[bold green]Solution found in {len(solution)-1} moves![/bold green]")
        for idx, state in enumerate(solution):
            console.print(f"[bold yellow]Step {idx}[/bold yellow]")
            render_board(state)
            sleep(0.5)
    else:
        console.print("[bold red]No solution found.[/bold red]")