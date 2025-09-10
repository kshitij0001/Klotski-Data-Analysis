# backend/scripts/board_generator.py

import sys
import os

# Add the backend directory to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from rich.console import Console
from backend.klotski.utils import Block

console = Console()


def render_klotski_board(blocks, rows=5, cols=4, return_string=False):
    """
    Render Klotski board either to console or as string.
    
    Args:
        blocks: List of Block objects
        rows: Number of rows (default 5)
        cols: Number of columns (default 4)
        return_string: If True, return as string instead of printing
    
    Returns:
        String representation if return_string=True, None otherwise
    """
    from rich.console import Console
    console = Console()

    # Build board index grid (None for empty cells)
    board = [[None for _ in range(cols)] for _ in range(rows)]
    for idx, block in enumerate(blocks):
        for r in range(block.row_pos, block.row_pos + block.num_rows):
            for c in range(block.col_pos, block.col_pos + block.num_cols):
                board[r][c] = idx

    # Helper to check same block
    def same_block(r1, c1, r2, c2):
        if not (0 <= r1 < rows and 0 <= c1 < cols and 0 <= r2 < rows
                and 0 <= c2 < cols):
            return False
        return board[r1][c1] == board[r2][c2] and board[r1][c1] is not None

    H, V = "─", "│"

    # Horizontal boundaries
    h = [[False for _ in range(cols)] for _ in range(rows + 1)]
    for c in range(cols):
        h[0][c] = True
        h[rows][c] = True
    for r_edge in range(1, rows):
        for c in range(cols):
            h[r_edge][c] = board[r_edge - 1][c] != board[r_edge][c]

    # Vertical boundaries
    v = [[False for _ in range(cols + 1)] for _ in range(rows)]
    for r in range(rows):
        v[r][0] = True
        v[r][cols] = True
        for c_edge in range(1, cols):
            v[r][c_edge] = board[r][c_edge - 1] != board[r][c_edge]

    # Junction char chooser
    def junction_char(up, down, left, right):
        if up and down and left and right: return "┼"
        if up and down and right and not left: return "├"
        if up and down and left and not right: return "┤"
        if left and right and down and not up: return "┬"
        if left and right and up and not down: return "┴"
        if down and right and not up and not left: return "┌"
        if down and left and not up and not right: return "┐"
        if up and right and not down and not left: return "└"
        if up and left and not down and not right: return "┘"
        if left and right and not up and not down: return H
        if up and down and not left and not right: return V
        if right: return H
        if left: return H
        if up: return V
        if down: return V
        return " "

    # Color chooser
    def cell_symbol(block_idx):
        if block_idx is None:
            return " X "
        if return_string:
            # Simple text for string output
            return "   "
        else:
            # Rich formatting for console output
            block = blocks[block_idx]
            if block.num_rows == 2 and block.num_cols == 2:
                return "[red1 on red1]   [/]"
            elif block.num_rows == 2 and block.num_cols == 1:
                return "[cyan1 on cyan1]   [/]"
            elif block.num_rows == 1 and block.num_cols == 2:
                return "[yellow1 on yellow1]   [/]"
            else:
                return "[green1 on green1]   [/]"

    # Render
    if return_string:
        output_lines = []
        
    for r_node in range(rows + 1):
        # Junction line
        line = ""
        for c_node in range(cols + 1):
            hor_left = (c_node > 0 and h[r_node][c_node - 1])
            hor_right = (c_node < cols and h[r_node][c_node])
            ver_up = (r_node > 0
                      and v[r_node - 1][c_node]) if r_node > 0 else False
            ver_down = (r_node < rows
                        and v[r_node][c_node]) if r_node < rows else False
            j = junction_char(ver_up, ver_down, hor_left, hor_right)
            line += j
            if c_node < cols:
                line += (H * 3) if h[r_node][c_node] else "   "
        
        if return_string:
            output_lines.append(line)
        else:
            console.print(line)

        # Content line
        if r_node < rows:
            content = ""
            for c in range(cols):
                content += (V if v[r_node][c] else " ")
                content += cell_symbol(board[r_node][c])
            content += (V if v[r_node][cols] else " ")
            
            if return_string:
                output_lines.append(content)
            else:
                console.print(content)
    
    if return_string:
        return "\n".join(output_lines)


# Run the renderer
if __name__ == "__main__":

    default_blocks = [
        Block(2, 2, 0, 1),
        Block(2, 1, 0, 0),
        Block(2, 1, 0, 3),
        Block(2, 1, 2, 0),
        Block(2, 1, 2, 3),
        Block(1, 1, 4, 0),
        Block(1, 1, 3, 1),
        Block(1, 1, 3, 2),
        Block(1, 1, 4, 3),
        Block(1, 2, 2, 1),
    ]

    console.print(
        "\n[bold green]Generated Klotski Board (Shape Only):[/bold green]")
    render_klotski_board(default_blocks)
