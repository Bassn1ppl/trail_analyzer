import sys
import traceback
import io
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
import gpxpy
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000

def parse_gpx(file_content):
    gpx = gpxpy.parse(file_content)
    data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                data.append({
                    "time": point.time,
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "ele": point.elevation or 0,
                })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df["prev_lat"] = df["lat"].shift(1)
    df["prev_lon"] = df["lon"].shift(1)
    df = df.iloc[1:].copy()
    
    window = min(21, len(df))
    if window % 2 == 0:
        window += 1
    if window > 3:
        df["ele"] = savgol_filter(df["ele"], window, 2)
    
    df["ele"] = df["ele"].fillna(0)
    
    df["dist_diff"] = haversine(
        df["prev_lat"].to_numpy(),
        df["prev_lon"].to_numpy(),
        df["lat"].to_numpy(),
        df["lon"].to_numpy(),
    )
    df["cum_dist"] = df["dist_diff"].cumsum()
    
    return df

# Load the GPX file
gpx_path = r"C:\Users\kmoll\OneDrive\Escritorio\trail_analyzer\training_files\Otomi 26 Training Block\60k-xanthe-utms-otomi-2025.gpx"

print(f"Loading GPX: {gpx_path}")
try:
    with open(gpx_path, "r", encoding="utf-8", errors="ignore") as f:
        content = io.StringIO(f.read())
    
    race_df = parse_gpx(content)
    print(f"Parsed GPX: {len(race_df)} points")
    print(f"Distance: {race_df['cum_dist'].max() / 1000:.2f} km")
    print(f"Elevation: {race_df['ele'].min():.0f}m to {race_df['ele'].max():.0f}m")
    
except Exception as e:
    print(f"ERROR parsing GPX: {e}")
    traceback.print_exc()
    sys.exit(1)

# Now test the chart rendering
app = QApplication(sys.argv)

try:
    print("Creating chart...")
    
    distances_km = (race_df["cum_dist"].to_numpy(dtype=float)) / 1000.0
    elevations_m = race_df["ele"].to_numpy(dtype=float)
    
    print(f"Points before downsampling: {len(distances_km)}")
    
    max_points = 1400
    num_points = len(distances_km)
    if num_points > max_points:
        sample_idx = np.linspace(0, num_points - 1, max_points).astype(int)
        distances_km = distances_km[sample_idx]
        elevations_m = elevations_m[sample_idx]
    
    print(f"Points after downsampling: {len(distances_km)}")
    
    # Setup axes
    x_axis = QValueAxis()
    x_axis.setTitleText("Distance (km)")
    x_axis.setLabelFormat("%.1f")
    x_axis.setRange(0.0, float(distances_km[-1]))
    
    y_axis = QValueAxis()
    y_axis.setTitleText("Elevation (m)")
    y_axis.setLabelFormat("%.0f")
    y_min = float(elevations_m.min())
    y_max = float(elevations_m.max())
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_axis.setRange(y_min, y_max)
    
    baseline = y_min
    print(f"Baseline: {baseline}, Y-axis range: {y_min} to {y_max}")
    
    def slope_to_color(slope_pct):
        if slope_pct >= 12:
            return QColor(186, 33, 33, 150)
        if slope_pct >= 6:
            return QColor(235, 96, 37, 145)
        if slope_pct >= 2:
            return QColor(245, 166, 35, 140)
        if slope_pct > -2:
            return QColor(82, 179, 88, 135)
        if slope_pct > -6:
            return QColor(55, 143, 201, 140)
        return QColor(42, 95, 184, 150)
    
    chart = QChart()
    print("Chart created")
    
    data_series = []
    
    legend_items = [
        ("≥12% (Steep Climb)", QColor(186, 33, 33, 150)),
        ("6-12% (Moderate Climb)", QColor(235, 96, 37, 145)),
        ("2-6% (Slight Climb)", QColor(245, 166, 35, 140)),
        ("-2 to 2% (Flat)", QColor(82, 179, 88, 135)),
        ("-6 to -2% (Slight Descent)", QColor(55, 143, 201, 140)),
        ("≤-6% (Steep Descent)", QColor(42, 95, 184, 150)),
    ]
    
    print("Adding legend items...")
    for label, color in legend_items:
        dummy = QLineSeries()
        dummy.setName(label)
        pen = QPen(color)
        pen.setWidth(6)
        dummy.setPen(pen)
        chart.addSeries(dummy)
    print(f"Added {len(legend_items)} legend items")
    
    chart.setTitle(
        f"Elevation Profile • {distances_km[-1]:.2f} km • "
        f"{elevations_m.min():.0f}m to {elevations_m.max():.0f}m"
    )
    print("Title set")
    
    chunk_size = 10
    data_series = []
    
    print("Building area chunks...")
    chunk_count = 0
    for start_idx in range(0, len(distances_km) - 1, chunk_size):
        end_idx = min(start_idx + chunk_size, len(distances_km) - 1)
        if end_idx <= start_idx:
            continue

        chunk_series = QLineSeries()

        for point_idx in range(start_idx, end_idx + 1):
            x_val = float(distances_km[point_idx])
            y_val = float(elevations_m[point_idx])
            chunk_series.append(x_val, y_val)

        dist_delta_m = float((distances_km[end_idx] - distances_km[start_idx]) * 1000.0)
        ele_delta_m = float(elevations_m[end_idx] - elevations_m[start_idx])
        slope_pct = (ele_delta_m / dist_delta_m) * 100.0 if dist_delta_m > 0 else 0.0

        pen = QPen(slope_to_color(slope_pct))
        pen.setWidth(5)
        chunk_series.setPen(pen)
        chart.addSeries(chunk_series)
        data_series.append(chunk_series)
        chunk_count += 1
    
    print(f"Added {chunk_count} colored line chunks")
    
    upper_line = QLineSeries()
    line_pen = QPen(QColor(35, 35, 35))
    line_pen.setWidth(2)
    upper_line.setPen(line_pen)

    print("Building upper line...")
    for distance_km, elevation_m in zip(distances_km, elevations_m):
        upper_line.append(float(distance_km), float(elevation_m))
    print(f"Upper line has {len(distances_km)} points")
    
    chart.addSeries(upper_line)
    data_series.append(upper_line)
    print(f"Added upper line, total data series: {len(data_series)}")
    
    chart.legend().setVisible(True)
    chart.legend().setAlignment(Qt.AlignRight)
    print("Legend configured")
    
    chart.addAxis(x_axis, Qt.AlignBottom)
    chart.addAxis(y_axis, Qt.AlignLeft)
    print("Axes added to chart")
    
    print("Attaching axes to data series...")
    for i, series in enumerate(data_series):
        print(f"  Attaching to series {i}...")
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
    print("All axes attached")
    
    print("Test passed! Chart rendering complete.")
    sys.exit(0)
    
except Exception as e:
    print(f"ERROR in chart: {e}")
    traceback.print_exc()
    sys.exit(1)
