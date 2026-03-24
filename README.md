# Trail Analyzer - Race Plan Optimizer

An intelligent race planning tool that analyzes trail running courses using GPX and FIT data to optimize race strategies, manage nutrition, and predict performance.

## Features

- **GPX/FIT File Analysis**: Parse and analyze trail running data
- **Route Visualization**: Interactive trail maps with elevation profiles
- **Segment Matching**: Intelligent matching of training segments to race courses
- **Performance Prediction**: Estimate race times based on training data
- **Nutrition Planning**: Optimize hydration and fuel strategy
- **Training Block Analysis**: Analyze training data for specific races

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/trail_analyzer.git
cd trail_analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run RacePlanOptimized.py
```

## Deployment

This app is deployed on Streamlit Cloud. Visit: [Your Streamlit App URL]

## Project Structure

```
trail_analyzer/
├── RacePlanOptimized.py      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── training_files/           # Sample GPX and FIT files
│   ├── 60k-xanthe-utms-otomi-2025.gpx
│   └── Otomi 26 Training Block/
│       ├── FITs/
│       └── GPXs/
└── README.md
```

## Usage

1. Upload your race course GPX file
2. Upload training data (GPX or FIT files)
3. Configure race parameters and nutrition strategy
4. View analysis and performance predictions

## Technologies Used

- **Streamlit**: Web app framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **GPXpy**: GPX file parsing
- **Fitparse**: FIT file parsing
- **NumPy/SciPy**: Scientific computing

## License

MIT License

## Author

Your Name
