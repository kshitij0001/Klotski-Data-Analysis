// backend/klotski_board/frontend/main.js
import { Streamlit } from "streamlit-component-lib"

const svgNS = "http://www.w3.org/2000/svg"
const boardEl = document.getElementById("board")

// Game state management
let gameState = {
  blocks: [],
  selectedBlockId: null,
  playerMoves: 0,
  minSteps: null,
  gameWon: false,
  winningRow: 3,
  winningCol: 1
}

// Constants
const ROWS = 5
const COLS = 4

function clearSvg(el){ while (el.firstChild) el.removeChild(el.firstChild) }
function add(el, tag, attrs={}){ const n=document.createElementNS(svgNS, tag); for(const[k,v] of Object.entries(attrs)) n.setAttribute(k,String(v)); el.appendChild(n); return n }

// Clone a block for state management
function cloneBlock(block) {
  return {
    id: block.id,
    num_rows: block.num_rows,
    num_cols: block.num_cols,
    row_pos: block.row_pos,
    col_pos: block.col_pos,
    is_goal: block.is_goal
  }
}

// Check if two positions overlap
function blocksOverlap(block1, block2) {
  if (block1.id === block2.id) return false
  
  const b1_right = block1.col_pos + block1.num_cols
  const b1_bottom = block1.row_pos + block1.num_rows
  const b2_right = block2.col_pos + block2.num_cols
  const b2_bottom = block2.row_pos + block2.num_rows
  
  return !(block1.col_pos >= b2_right || block2.col_pos >= b1_right ||
           block1.row_pos >= b2_bottom || block2.row_pos >= b1_bottom)
}

// Check if a block is within board bounds
function isWithinBounds(block) {
  return block.row_pos >= 0 && 
         block.col_pos >= 0 && 
         block.row_pos + block.num_rows <= ROWS && 
         block.col_pos + block.num_cols <= COLS
}

// Check if a move is valid
function isValidMove(blockId, newRow, newCol) {
  const block = gameState.blocks.find(b => b.id === blockId)
  if (!block) return false
  
  // Create a test block at the new position
  const testBlock = cloneBlock(block)
  testBlock.row_pos = newRow
  testBlock.col_pos = newCol
  
  // Check bounds
  if (!isWithinBounds(testBlock)) return false
  
  // Check collisions with other blocks (excluding itself)
  for (const otherBlock of gameState.blocks) {
    if (otherBlock.id !== blockId && blocksOverlap(testBlock, otherBlock)) {
      return false
    }
  }
  
  return true
}

// Get all valid move positions for a block (proper sliding logic)
function getValidMoves(blockId) {
  const block = gameState.blocks.find(b => b.id === blockId)
  if (!block) return []
  
  const validMoves = []
  const directions = [
    { row: -1, col: 0, name: 'up' },    // up
    { row: 1, col: 0, name: 'down' },   // down
    { row: 0, col: -1, name: 'left' },  // left
    { row: 0, col: 1, name: 'right' }   // right
  ]
  
  // For each direction, find how far the block can slide
  for (const dir of directions) {
    let steps = 1
    let canMove = true
    
    while (canMove && steps <= 3) { // Max 3 steps to prevent infinite loops
      const newRow = block.row_pos + (dir.row * steps)
      const newCol = block.col_pos + (dir.col * steps)
      
      if (isValidMove(blockId, newRow, newCol)) {
        validMoves.push({ 
          row: newRow, 
          col: newCol, 
          direction: dir.name,
          steps: steps 
        })
        steps++
      } else {
        canMove = false
      }
    }
  }
  
  return validMoves
}

// Move a block to a new position
function moveBlock(blockId, newRow, newCol) {
  if (!isValidMove(blockId, newRow, newCol)) return false
  
  const block = gameState.blocks.find(b => b.id === blockId)
  if (!block) return false
  
  // Only count as a move if the position actually changes
  if (block.row_pos !== newRow || block.col_pos !== newCol) {
    block.row_pos = newRow
    block.col_pos = newCol
    gameState.playerMoves++
    
    // Check win condition
    checkWinCondition()
    
    // Notify Streamlit of game state update
    notifyGameStateUpdate()
    
    return true
  }
  
  return false
}

// Check if the puzzle is solved
function checkWinCondition() {
  const goalBlock = gameState.blocks.find(b => b.is_goal || (b.num_rows === 2 && b.num_cols === 2))
  if (goalBlock && 
      goalBlock.row_pos === gameState.winningRow && 
      goalBlock.col_pos === gameState.winningCol) {
    gameState.gameWon = true
    showWinMessage()
  }
}

// Show win message
function showWinMessage() {
  Streamlit.setComponentValue({
    type: "win",
    playerMoves: gameState.playerMoves,
    minSteps: gameState.minSteps
  })
}

// Find the best move toward a target position
function findBestMoveToward(blockId, targetRow, targetCol) {
  const block = gameState.blocks.find(b => b.id === blockId)
  if (!block) return null
  
  const validMoves = getValidMoves(blockId)
  if (validMoves.length === 0) return null
  
  // Calculate click direction relative to block center
  const blockCenterRow = block.row_pos + (block.num_rows / 2)
  const blockCenterCol = block.col_pos + (block.num_cols / 2)
  
  const deltaRow = targetRow - blockCenterRow
  const deltaCol = targetCol - blockCenterCol
  
  // Determine intended direction based on which delta is larger
  let intendedDirection = null
  if (Math.abs(deltaRow) > Math.abs(deltaCol)) {
    intendedDirection = deltaRow > 0 ? 'down' : 'up'
  } else {
    intendedDirection = deltaCol > 0 ? 'right' : 'left'
  }
  
  // Find the best move in the intended direction
  let bestMove = null
  let maxSteps = 0
  
  for (const move of validMoves) {
    if (move.direction === intendedDirection && move.steps > maxSteps) {
      maxSteps = move.steps
      bestMove = move
    }
  }
  
  // If no move in intended direction, fall back to closest move
  if (!bestMove) {
    let bestDistance = Infinity
    for (const move of validMoves) {
      const distance = Math.abs(move.row - targetRow) + Math.abs(move.col - targetCol)
      if (distance < bestDistance) {
        bestDistance = distance
        bestMove = move
      }
    }
  }
  
  return bestMove
}

// Notify Streamlit of game state updates
function notifyGameStateUpdate() {
  Streamlit.setComponentValue({
    type: "gameUpdate",
    playerMoves: gameState.playerMoves,
    gameWon: gameState.gameWon,
    blocks: gameState.blocks
  })
}

// Reset game state
function resetGame() {
  gameState.selectedBlockId = null
  gameState.playerMoves = 0
  gameState.gameWon = false
  render()
  notifyGameStateUpdate()
}

// Get API base URL based on environment
function getApiBaseUrl() {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000'  // Local development
  } else {
    // For Replit or other cloud environments, use current hostname with port 8000
    const hostname = window.location.hostname.split(':')[0]  // Remove any existing port
    return `https://${hostname}:8000`
  }
}

// Request new puzzle from Streamlit
function loadNewPuzzle() {
  Streamlit.setComponentValue({
    type: "newPuzzle"
  })
}

// Request solution from Streamlit
function getSolution() {
  // Convert our blocks to the format expected by Streamlit
  const blockData = gameState.blocks.map(block => ({
    num_rows: block.num_rows,
    num_cols: block.num_cols,
    row_pos: block.row_pos,
    col_pos: block.col_pos
  }))
  
  Streamlit.setComponentValue({
    type: "getSolution",
    blocks: blockData,
    winningRow: gameState.winningRow,
    winningCol: gameState.winningCol
  })
}

// Render the game board
function render() {
  clearSvg(boardEl)
  boardEl.setAttribute("viewBox", `0 0 ${COLS} ${ROWS}`)

  // Grid lines
  for(let x=0; x<=COLS; x++) add(boardEl,"line",{x1:x,y1:0,x2:x,y2:ROWS,class:"grid-line"})
  for(let y=0; y<=ROWS; y++) add(boardEl,"line",{x1:0,y1:y,x2:COLS,y2:y,class:"grid-line"})

  // Highlight valid moves for selected block
  if (gameState.selectedBlockId !== null) {
    const validMoves = getValidMoves(gameState.selectedBlockId)
    for (const move of validMoves) {
      const block = gameState.blocks.find(b => b.id === gameState.selectedBlockId)
      if (block) {
        add(boardEl,"rect",{
          x: move.col,
          y: move.row,
          width: block.num_cols,
          height: block.num_rows,
          fill: "rgba(76, 175, 80, 0.3)",
          stroke: "rgba(76, 175, 80, 0.8)",
          "stroke-width": 0.05,
          rx: 0.1,
          ry: 0.1,
          class: "highlight-square"
        })
      }
    }
  }

  // Background click handler
  const backgroundRect = add(boardEl,"rect",{
    x: 0, y: 0, width: COLS, height: ROWS,
    fill: "transparent", class: "board-background"
  })
  
  backgroundRect.addEventListener("click",(e)=>{
    e.stopPropagation()
    if (gameState.selectedBlockId !== null && !gameState.gameWon) {
      const rect = boardEl.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width * COLS
      const y = (e.clientY - rect.top) / rect.height * ROWS
      
      const gridX = Math.floor(x)
      const gridY = Math.floor(y)
      
      // Find the best valid move toward the clicked position
      const targetMove = findBestMoveToward(gameState.selectedBlockId, gridY, gridX)
      if (targetMove && moveBlock(gameState.selectedBlockId, targetMove.row, targetMove.col)) {
        gameState.selectedBlockId = null
        render()
      }
    }
  })

  // Render blocks
  for(const block of gameState.blocks){
    let blockClass = "klotski-block "
    
    // Color by size
    if (block.num_rows === 1 && block.num_cols === 1) {
      blockClass += "block-1x1"
    } else if (block.num_rows === 1 && block.num_cols === 2) {
      blockClass += "block-1x2"
    } else if (block.num_rows === 2 && block.num_cols === 1) {
      blockClass += "block-2x1"
    } else if (block.num_rows === 2 && block.num_cols === 2) {
      blockClass += "block-2x2"
    }
    
    // Selection highlight
    if (gameState.selectedBlockId === block.id) {
      blockClass += " selected"
    }
    
    const rect = add(boardEl,"rect",{
      x: block.col_pos, y: block.row_pos,
      width: block.num_cols, height: block.num_rows,
      rx: 0.12, ry: 0.12, class: blockClass
    })
    
    rect.addEventListener("click",(e)=>{
      e.stopPropagation()
      if (!gameState.gameWon) {
        gameState.selectedBlockId = gameState.selectedBlockId === block.id ? null : block.id
        render()
      }
    })
  }

  Streamlit.setFrameHeight()
}

// Handle Streamlit render events
function onRender(event) {
  const args = event.detail.args
  
  // Only handle puzzle data updates from Streamlit
  if (args.blocks) {
    gameState.blocks = args.blocks.map(b => cloneBlock(b))
    gameState.minSteps = args.minSteps || null
    gameState.selectedBlockId = null
    gameState.playerMoves = 0
    gameState.gameWon = false
    render()
  }
}

// Initialize
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
Streamlit.setFrameHeight()