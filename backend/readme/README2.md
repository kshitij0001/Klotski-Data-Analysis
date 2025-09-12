
# 🧩 Klotski Puzzle Solver

A full-stack Klotski puzzle solver application that combines classical algorithms with modern web technologies. Features a Python backend with sophisticated graph search algorithms and an interactive Streamlit frontend.

## 🚀 Features

- **Intelligent Solver**: Uses BFS and A* algorithms to find optimal solutions
- **Random Puzzle Generation**: Creates solvable puzzles with configurable difficulty
- **Interactive UI**: Click-to-select and drag-to-move gameplay with visual animations
- **Solution Playback**: Step-by-step solution visualization with auto-play
- **Multiple Interfaces**: Both Streamlit dashboard and FastAPI endpoints
- **Real-time Analytics**: Performance metrics and solving statistics

## 🏗️ Project Structure

```
backend/
├── klotski/           # Core puzzle logic and solver
├── api/               # FastAPI REST endpoints  
├── dashboard/         # Streamlit web interface
├── klotski_board/     # Custom Streamlit component
├── scripts/           # Utilities and data generation
└── tests/             # Unit tests
```

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.8+
- Node.js 16+ (for building the frontend component)

### Step 1: Install Dependencies

The project uses Python's built-in dependency management. All required packages are automatically installed when you run the application.

```bash
# Core Python dependencies will be auto-installed:
# streamlit, fastapi, uvicorn, pandas, numpy, networkx, etc.
```

### Step 2: Build the Frontend Component

The Klotski board uses a custom Streamlit component built with JavaScript:

```bash
cd backend/klotski_board/frontend
npm install
npm run build
cd ../../../
```

### Step 3: Run the Application

#### Option A: Streamlit Dashboard (Recommended)

```bash
cd backend
streamlit run dashboard/app.py
```

Access the dashboard at: `http://localhost:8501`

#### Option B: FastAPI Server

```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Access the API documentation at: `http://localhost:8000/docs`

#### Option C: CLI Solver

```bash
cd backend
python -m klotski --preset
```

## 🎮 How to Use

### Interactive Dashboard

1. **Generate Puzzle**: Click "🔀 New Puzzle" for a random solvable configuration
2. **Solve**: Click "🧠 Solve" to find the optimal solution path
3. **Navigate**: Use "⏮️ Prev" / "⏭️ Next" to step through solutions
4. **Auto-play**: Click "▶️ Auto Play" for animated solution playback
5. **Manual Play**: Click blocks to select them, then click edges to move

### API Endpoints

- `POST /solve` - Submit a puzzle configuration for solving
- `GET /random` - Generate a random solvable puzzle
- `GET /health` - Check server status

## 🔧 Configuration

### Puzzle Parameters

- **Board Size**: 5 rows × 4 columns (standard Klotski)
- **Goal Position**: Row 3, Column 1 (configurable)
- **Block Types**: 
  - 🟥 2×2 (Goal block - red)
  - 🟦 2×1 (Vertical blocks - blue)  
  - 🟨 1×2 (Horizontal blocks - yellow)
  - 🟩 1×1 (Small blocks - green)

### Development Modes

Set `_RELEASE = False` in `backend/klotski_board/__init__.py` to use the Vite dev server during component development.

## 🧪 Testing

```bash
cd backend
python -m pytest tests/
```

## 📊 Performance

- **Average Solve Time**: ~2-5 seconds for standard puzzles
- **Solution Quality**: Guaranteed optimal (shortest path)
- **Memory Usage**: Efficient state representation with duplicate detection
- **Scalability**: Handles complex 30+ move solutions

## 🎯 Algorithm Details

### Solver Implementation

- **BFS (Breadth-First Search)**: Guarantees optimal solutions
- **A* Search**: Heuristic-based optimization for faster solving
- **State Deduplication**: Prevents cycles and reduces memory usage
- **Move Generation**: Intelligent block movement with collision detection

### Data Structures

- **Board Representation**: 2D grid with block position tracking
- **State Hashing**: Efficient puzzle state comparison
- **Solution Tree**: Path reconstruction for step-by-step playback

## 🚀 Deployment on Replit

### Quick Deploy

1. Fork this repository on Replit
2. The project will automatically install dependencies
3. Click "Run" to start the Streamlit dashboard
4. Access your deployed app through Replit's provided URL

### Custom Configuration

The project includes pre-configured workflows:
- **Main App**: Runs the Streamlit dashboard
- **API Server**: Starts the FastAPI backend
- **Component Build**: Rebuilds the frontend component

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**"Component not loading"**
- Ensure the frontend component is built: `cd backend/klotski_board/frontend && npm run build`

**"No solution found"**  
- Try generating a new puzzle - some configurations may be unsolvable

**"Port already in use"**
- Change the port in the run command: `--server.port 5001`

### Debug Mode

Enable verbose logging by setting environment variables:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
export UVICORN_LOG_LEVEL=debug
```

---

**Made with ❤️ using Python, Streamlit, and modern web technologies**
