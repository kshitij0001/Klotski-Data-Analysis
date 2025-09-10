# backend/api/main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import sys
import os

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from backend.klotski.solver import Solver
from backend.klotski.utils import Block
from backend.scripts.generate_random_klotski_board import generate_random_klotski_board

app = FastAPI(
    title="Klotski Puzzle Solver API",
    description="API for generating and solving Klotski puzzles with classical algorithms.",
    version="1.0.0"
)

# --- Models ---
class BlockModel(BaseModel):
    num_rows: int
    num_cols: int
    row_pos: int
    col_pos: int

class SolveRequest(BaseModel):
    blocks: List[BlockModel]
    winning_position: dict = {"row": 3, "col": 1}


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Klotski Solver API</title></head>
        <body style="font-family:sans-serif; max-width:600px; margin:2rem auto;">
            <h1>🧩 Klotski Solver API</h1>
            <p>This API lets you generate and solve <b>Klotski sliding block puzzles</b>.</p>
            <ul>
                <li>Visit <a href='/docs'>/docs</a> for interactive API docs.</li>
                <li>POST to <code>/solve</code> with a puzzle definition to get a solution path.</li>
                <li>Try <a href='/sample'>/sample</a> for a random puzzle demo.</li>
            </ul>
        </body>
    </html>
    """


@app.post("/solve")
async def solve_puzzle(request: SolveRequest):
    blocks = [Block(b.num_rows, b.num_cols, b.row_pos, b.col_pos) for b in request.blocks]
    solver = Solver(blocks, request.winning_position["row"], request.winning_position["col"])
    solver.solve()
    solution = solver.get_solution_boards()

    return JSONResponse({
        "steps": len(solution) - 1 if solution else None,
        "solution": [
            [{"rows": b.num_rows, "cols": b.num_cols, "row": b.row_pos, "col": b.col_pos} for b in state]
            for state in (solution or [])
        ]
    })


@app.get("/sample")
async def sample_puzzle():
    blocks, _ = generate_random_klotski_board()
    solver = Solver(blocks, 3, 1)
    solver.solve()
    solution = solver.get_solution_boards()

    return {
        "initial": [{"rows": b.num_rows, "cols": b.num_cols, "row": b.row_pos, "col": b.col_pos} for b in blocks],
        "steps": len(solution) - 1 if solution else None,
        "solution": [
            [{"rows": b.num_rows, "cols": b.num_cols, "row": b.row_pos, "col": b.col_pos} for b in state]
            for state in (solution or [])
        ]
    }
