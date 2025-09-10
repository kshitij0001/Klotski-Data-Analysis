# # backend/klotski/board.py

from typing import List, Optional, Set
from .utils import Block, Dir, Move, clone_block, move_block, equivalent_blocks, opposite_dir, opposite_dirs

class Board:
    """
    Represents the Klotski board, manages block placement, move generation, and goal checking.
    """

    ROWS = 5
    COLS = 4

    def __init__(self, blocks: List[Block], winning_row: int, winning_col: int):
        """
        Initializes the board with blocks and the winning position.

        Args:
            blocks (List[Block]): List of blocks to place on the board.
            winning_row (int): Row index of the winning position (top-left of 2x2 area).
            winning_col (int): Column index of the winning position (top-left of 2x2 area).
        """
        self.blocks: List[Block] = blocks
        self.winning_row: int = winning_row
        self.winning_col: int = winning_col
        self._cells: List[List[Optional[Block]]] = [
            [None for _ in range(self.COLS)] for _ in range(self.ROWS)
        ]
        self._insert_blocks()

    def _insert_blocks(self) -> None:
        """
        Places all blocks on the board, ensuring validity.
        Raises ValueError if the configuration is invalid.
        """
        four_block_found = False
        covered_cells = 0
        for block in self.blocks:
            # Check if block is within bounds
            if (block.row_pos < 0 or 
                block.col_pos < 0 or 
                block.row_pos + block.num_rows > self.ROWS or 
                block.col_pos + block.num_cols > self.COLS):
                raise ValueError(f"Block is out of bounds: {block}")
    
            if block.num_rows * block.num_cols == 4:
                if not four_block_found:
                    four_block_found = True
                else:
                    raise ValueError("There must be exactly one block of size 4.")
            for i in range(block.row_pos, block.row_pos + block.num_rows):
                for j in range(block.col_pos, block.col_pos + block.num_cols):
                    if not (0 <= i < self.ROWS and 0 <= j < self.COLS):
                        raise ValueError(f"Block at ({block.row_pos},{block.col_pos}) with size {block.num_rows}x{block.num_cols} is out of bounds.")
                    if self._cells[i][j] is None:
                        self._cells[i][j] = block
                        covered_cells += 1
                    else:
                        raise ValueError("Invalid block positioning: overlapping blocks.")
        if covered_cells != 18:
            raise ValueError("There must be exactly two free spaces.")
        if not four_block_found:
            raise ValueError("There must be exactly one block of size 4.")

    def _current_dirs(self, block: Block) -> List[Dir]:
        """
        Returns a list of valid directions the given block can move.

        Args:
            block (Block): The block to check.

        Returns:
            List[Dir]: List of possible movement directions.
        """
        dirs = []
        left = right = up = down = True

        # Check horizontal moves
        for row in range(block.row_pos, block.row_pos + block.num_rows):
            col = block.col_pos
            if left and (col < 1 or self._cells[row][col - 1] is not None):
                left = False
            col_r = block.col_pos + block.num_cols - 1
            if right and (col_r > self.COLS - 2 or self._cells[row][col_r + 1] is not None):
                right = False

        # Check vertical moves
        for col in range(block.col_pos, block.col_pos + block.num_cols):
            row = block.row_pos
            if up and (row < 1 or self._cells[row - 1][col] is not None):
                up = False
            row_b = block.row_pos + block.num_rows - 1
            if down and (row_b > self.ROWS - 2 or self._cells[row_b + 1][col] is not None):
                down = False

        if left:
            dirs.append(Dir.LEFT)
        if right:
            dirs.append(Dir.RIGHT)
        if up:
            dirs.append(Dir.UP)
        if down:
            dirs.append(Dir.DOWN)
        return dirs

    def _make_move(self, block: Block, dirs: List[Dir]) -> None:
        """
        Moves the block in-place and updates the board cells accordingly.

        Args:
            block (Block): The block to move.
            dirs (List[Dir]): List of directions to move.
        """
        num_rows = block.num_rows
        num_cols = block.num_cols
        for dir in dirs:
            init_row = block.row_pos
            init_col = block.col_pos
            move_block(block, [dir])
            # Update cells for left/up moves
            if dir in (Dir.LEFT, Dir.UP):
                for i in range(init_row, init_row + num_rows):
                    for j in range(init_col, init_col + num_cols):
                        if dir == Dir.LEFT:
                            self._cells[i][j - 1] = block
                        else:
                            self._cells[i - 1][j] = block
                        self._cells[i][j] = None
            # Update cells for right/down moves
            else:
                for i in range(init_row + num_rows - 1, init_row - 1, -1):
                    for j in range(init_col + num_cols - 1, init_col - 1, -1):
                        if dir == Dir.RIGHT:
                            self._cells[i][j + 1] = block
                        else:
                            self._cells[i + 1][j] = block
                        self._cells[i][j] = None

    def _find_moves(self, block: Block) -> List[List[Dir]]:
        """
        Finds all valid move paths (single and double moves) for a block.

        Args:
            block (Block): The block to check.

        Returns:
            List[List[Dir]]: List of direction sequences (each a valid move).
        """
        res = []
        for init_dir in self._current_dirs(block):
            res.append([init_dir])
            self._make_move(block, [init_dir])
            for next_dir in self._current_dirs(block):
                if next_dir != opposite_dir(init_dir):
                    res.append([init_dir, next_dir])
            self._make_move(block, opposite_dirs([init_dir]))  # Undo move
        return res

    def get_moves(self) -> List[Move]:
        """
        Returns all valid moves for all blocks on the board.

        Returns:
            List[Move]: List of possible moves.
        """
        moves = []
        for block in self.blocks:
            for dirs in self._find_moves(block):
                new_block = clone_block(block)
                moves.append(Move(block=new_block, dirs=dirs))
        return moves

    def get_hash(self) -> str:
        """
        Returns a string hash representing the board state.

        Returns:
            str: Board hash.
        """
        res = ""
        for i in range(self.ROWS):
            for j in range(self.COLS):
                block = self._cells[i][j]
                if block is None:
                    res += "0"
                elif block.num_rows * block.num_cols == 2:
                    res += "2H" if block.num_rows == 1 else "2V"
                else:
                    res += str(block.num_rows * block.num_cols)
        return res

    def is_solved(self) -> bool:
        """
        Checks if the board is in a solved state.

        Returns:
            bool: True if solved, False otherwise.
        """
        block = self._cells[self.winning_row][self.winning_col]
        return (
            block is not None and
            block.row_pos == self.winning_row and
            block.col_pos == self.winning_col and
            block.num_rows * block.num_cols == 4
        )