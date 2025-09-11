# mirror_checker.py
"""
Standalone utility for checking horizontal (left-right) mirror symmetry in Klotski puzzle boards.
"""

import sys
import os

# Add the backend directory to sys.path to import Block class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

try:
    from klotski.utils import Block
except ImportError:
    # Fallback Block class if import fails
    class Block:
        def __init__(self, num_rows, num_cols, row_pos, col_pos):
            self.num_rows = num_rows
            self.num_cols = num_cols
            self.row_pos = row_pos
            self.col_pos = col_pos


def horizontal_mirror(blocks, board_cols=4):
    """
    Create horizontal mirror (left-right flip) of a board.
    
    Args:
        blocks: List of Block objects
        board_cols: Width of the board (default 4 for Klotski)
    
    Returns:
        List of mirrored Block objects
    """
    mirrored_blocks = []
    for block in blocks:
        # Calculate new column position: board_cols - original_col - block_width
        new_col = board_cols - block.col_pos - block.num_cols
        mirrored_blocks.append(Block(
            num_rows=block.num_rows,
            num_cols=block.num_cols,
            row_pos=block.row_pos,  # Row stays the same for horizontal mirror
            col_pos=new_col
        ))
    return mirrored_blocks


def blocks_to_canonical_form(blocks):
    """
    Convert blocks to a canonical form for comparison.
    Sorts by position and dimensions for consistent comparison.
    
    Args:
        blocks: List of Block objects
    
    Returns:
        Sorted list of tuples (row, col, height, width)
    """
    return sorted([
        (block.row_pos, block.col_pos, block.num_rows, block.num_cols)
        for block in blocks
    ])


def boards_are_equivalent(blocks1, blocks2):
    """
    Check if two board states are equivalent (same block positions).
    
    Args:
        blocks1, blocks2: Lists of Block objects
    
    Returns:
        bool: True if boards are equivalent
    """
    if len(blocks1) != len(blocks2):
        return False
    
    canonical1 = blocks_to_canonical_form(blocks1)
    canonical2 = blocks_to_canonical_form(blocks2)
    
    return canonical1 == canonical2


def is_horizontal_mirror(blocks1, blocks2, board_cols=4):
    """
    Check if blocks2 is a horizontal mirror of blocks1.
    
    Args:
        blocks1, blocks2: Lists of Block objects to compare
        board_cols: Width of the board
    
    Returns:
        bool: True if blocks2 is horizontal mirror of blocks1
    """
    mirrored_blocks1 = horizontal_mirror(blocks1, board_cols)
    return boards_are_equivalent(mirrored_blocks1, blocks2)


def has_horizontal_symmetry(blocks, board_cols=4):
    """
    Check if a board is horizontally symmetric (mirrors itself).
    
    Args:
        blocks: List of Block objects
        board_cols: Width of the board
    
    Returns:
        bool: True if board is horizontally symmetric
    """
    mirrored_blocks = horizontal_mirror(blocks, board_cols)
    return boards_are_equivalent(blocks, mirrored_blocks)


def print_board_comparison(blocks1, blocks2, board_rows=5, board_cols=4):
    """
    Print visual comparison of two boards side by side.
    
    Args:
        blocks1, blocks2: Lists of Block objects
        board_rows, board_cols: Board dimensions
    """
    def blocks_to_grid(blocks):
        grid = [['.' for _ in range(board_cols)] for _ in range(board_rows)]
        for i, block in enumerate(blocks):
            symbol = str(i) if i < 10 else chr(ord('A') + i - 10)
            for r in range(block.row_pos, block.row_pos + block.num_rows):
                for c in range(block.col_pos, block.col_pos + block.num_cols):
                    if 0 <= r < board_rows and 0 <= c < board_cols:
                        grid[r][c] = symbol
        return grid
    
    grid1 = blocks_to_grid(blocks1)
    grid2 = blocks_to_grid(blocks2)
    
    print("Board 1:" + " " * 10 + "Board 2:")
    for r in range(board_rows):
        row1 = " ".join(grid1[r])
        row2 = " ".join(grid2[r])
        print(f"{row1}    {row2}")


def find_mirror_pairs_in_list(boards_list, board_cols=4):
    """
    Find all pairs of boards that are horizontal mirrors of each other.
    
    Args:
        boards_list: List of board states (each is a list of Block objects)
        board_cols: Width of the board
    
    Returns:
        List of tuples (index1, index2) indicating mirror pairs
    """
    mirror_pairs = []
    
    for i in range(len(boards_list)):
        for j in range(i + 1, len(boards_list)):
            if is_horizontal_mirror(boards_list[i], boards_list[j], board_cols):
                mirror_pairs.append((i, j))
    
    return mirror_pairs


def remove_mirror_duplicates(boards_list, board_cols=4, keep_first=True):
    """
    Remove boards that are horizontal mirrors of earlier boards in the list.
    
    Args:
        boards_list: List of board states
        board_cols: Width of the board
        keep_first: If True, keep the first occurrence; if False, keep the last
    
    Returns:
        List of unique boards (no horizontal mirrors)
    """
    unique_boards = []
    
    for current_board in boards_list:
        is_duplicate = False
        
        for unique_board in unique_boards:
            if is_horizontal_mirror(current_board, unique_board, board_cols):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_boards.append(current_board)
    
    return unique_boards


# Example usage and testing
if __name__ == "__main__":
    # Test with some example blocks
    print("Mirror Checker Test")
    print("=" * 50)
    
    # Create a simple test board
    test_blocks = [
        Block(2, 2, 0, 1),  # 2x2 block at (0,1) - goal block
        Block(1, 1, 2, 0),  # 1x1 block at (2,0)
        Block(1, 1, 2, 3),  # 1x1 block at (2,3) - mirror of above
        Block(2, 1, 0, 0),  # 2x1 block at (0,0)
        Block(2, 1, 0, 3),  # 2x1 block at (0,3) - mirror of above
    ]
    
    # Check if this board has horizontal symmetry
    is_symmetric = has_horizontal_symmetry(test_blocks)
    print(f"Test board is horizontally symmetric: {is_symmetric}")
    
    # Create the horizontal mirror
    mirrored_blocks = horizontal_mirror(test_blocks)
    
    print("\nOriginal vs Mirrored Board:")
    print_board_comparison(test_blocks, mirrored_blocks)
    
    # Verify they are mirrors
    is_mirror = is_horizontal_mirror(test_blocks, mirrored_blocks)
    print(f"\nBoards are horizontal mirrors: {is_mirror}")
    
    print("\nExample completed! Use this file to check mirror symmetry in your Klotski puzzles.")