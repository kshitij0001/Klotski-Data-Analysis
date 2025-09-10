# backend/klotski_board/__init__.py
import os
import streamlit.components.v1 as components  # official API

_RELEASE = True  # switch to True after building the frontend

if not _RELEASE:
    # Vite dev server URL (adjust if different)
    _component = components.declare_component(
        "klotski_board", url="http://localhost:5173"
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "dist")  # vite build output
    _component = components.declare_component(
        "klotski_board", path=build_dir
    )

def klotski_board(
    blocks,
    rows,
    cols,
    winningRow=None,         
    winningCol=None,         
    highlights=None,
    selected=None,
    minSteps=None,
    command=None,
    key=None,
    default=None,
):
    return _component(
        blocks=blocks,
        rows=rows,
        cols=cols,
        winningRow=winningRow,    
        winningCol=winningCol,    
        highlights=highlights or [],
        selected=selected,
        minSteps=minSteps,
        command=command,
        key=key,
        default=default,
    )