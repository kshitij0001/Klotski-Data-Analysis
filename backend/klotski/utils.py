# # backend/klotski/utils.py

from enum import Enum, auto
from typing import List
from dataclasses import dataclass, field
import copy

class Dir(Enum):
    """
    Enumeration for possible movement directions in Klotski.
    """
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()

def opposite_dir(dir: 'Dir') -> 'Dir':
    """
    Returns the opposite direction of the given direction.

    Args:
        dir (Dir): The direction to invert.

    Returns:
        Dir: The opposite direction.

    Raises:
        ValueError: If the direction is unknown.
    """
    if dir == Dir.LEFT:
        return Dir.RIGHT
    elif dir == Dir.RIGHT:
        return Dir.LEFT
    elif dir == Dir.UP:
        return Dir.DOWN
    elif dir == Dir.DOWN:
        return Dir.UP
    else:
        raise ValueError(f"Unknown direction: {dir}")

def opposite_dirs(dirs: List['Dir']) -> List['Dir']:
    """
    Returns a reversed list of the opposites of the given directions.

    Args:
        dirs (List[Dir]): List of directions.

    Returns:
        List[Dir]: Reversed list of opposite directions.
    """
    return [opposite_dir(d) for d in reversed(dirs)]

@dataclass(frozen=False)
class Block:
    """
    Represents a block on the Klotski board.

    Attributes:
        num_rows (int): Number of rows the block occupies.
        num_cols (int): Number of columns the block occupies.
        row_pos (int): Top-left row position of the block.
        col_pos (int): Top-left column position of the block.
    """
    num_rows: int
    num_cols: int
    row_pos: int
    col_pos: int

def clone_block(block: Block) -> Block:
    """
    Returns a deep copy of the given block.

    Args:
        block (Block): The block to clone.

    Returns:
        Block: A new block with the same attributes.
    """
    return copy.deepcopy(block)

@dataclass
class Move:
    """
    Represents a move in Klotski.

    Attributes:
        block (Block): The block being moved.
        dirs (List[Dir]): List of directions for the move.
    """
    block: Block
    dirs: List[Dir] = field(default_factory=list)

def move_block(block: Block, dirs: List[Dir]) -> None:
    """
    Mutates the block in-place by moving it according to the given directions.

    Args:
        block (Block): The block to move.
        dirs (List[Dir]): List of directions to move the block.
    """
    for dir in dirs:
        if dir == Dir.LEFT:
            block.col_pos -= 1
        elif dir == Dir.RIGHT:
            block.col_pos += 1
        elif dir == Dir.UP:
            block.row_pos -= 1
        elif dir == Dir.DOWN:
            block.row_pos += 1
        else:
            raise ValueError(f"Unknown direction: {dir}")

def equivalent_blocks(b1: Block, b2: Block) -> bool:
    """
    Checks if two blocks are equivalent in position and size.

    Args:
        b1 (Block): First block.
        b2 (Block): Second block.

    Returns:
        bool: True if blocks are equivalent, False otherwise.
    """
    return (
        b1.row_pos == b2.row_pos and
        b1.col_pos == b2.col_pos and
        b1.num_rows == b2.num_rows and
        b1.num_cols == b2.num_cols
    )