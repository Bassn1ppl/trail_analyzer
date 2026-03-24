import streamlit as st
import pandas as pd
import plotly.express as px
import gpxpy
import numpy as np
from scipy.signal import savgol_filter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Race Planner Pro - Optimized", page_icon="🏁", layout="wide")

# --- UTILITY FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000

def seconds_to_hms(seconds):
    """Convert seconds to HMS string format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02}:{s:02}"

# --- LOGIC CLASS ---
class RaceLogic:
    def __init__(self):
        self.nutrition = {"water": 20, "carbs": 60}
        # Segment matching tolerances are now auto-optimized (5-20%)
        self.min_search_points = 50
        self.max_search_points = 150

    def parse_gpx(self, file_content):
        """Parse GPX file with improved error handling."""
        try:
            gpx = gpxpy.parse(file_content)
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        data.append({
                            'time': point.time,
                            'lat': point.latitude,
                            'lon': point.longitude,
                            'ele': point.elevation or 0
                        })

            if not data:
                return None

            df = pd.DataFrame(data)
            df['prev_lat'] = df['lat'].shift(1)
            df['prev_lon'] = df['lon'].shift(1)
            df = df.iloc[1:].copy()

            # Smooth elevation with adaptive window
            window = min(21, len(df))
            if window % 2 == 0:
                window += 1
            if window > 3:
                df['ele'] = savgol_filter(df['ele'], window, 2)

            df['ele'] = df['ele'].fillna(0)

            # Vectorized distance calculation
            df['dist_diff'] = haversine(
                df['prev_lat'].to_numpy(),
                df['prev_lon'].to_numpy(),
                df['lat'].to_numpy(),
                df['lon'].to_numpy()
            )
            df['cum_dist'] = df['dist_diff'].cumsum()

            # Elevation calculations
            df['ele_diff'] = df['ele'].diff().fillna(0)
            df['cum_gain'] = df['ele_diff'].clip(lower=0).cumsum()
            df['cum_loss'] = df['ele_diff'].clip(upper=0).abs().cumsum()

            return df
        except Exception as e:
            st.error(f"Error parsing GPX: {e}")
            return None

    def auto_segment(self, df):
        """Auto-segment course by terrain type with optimizations."""
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        sample_dist = 100
        df['chunk_id'] = (df['cum_dist'] // sample_dist).astype(int)

        chunks = df.groupby('chunk_id').agg({
            'cum_dist': 'last',
            'ele': ['first', 'last']
        }).reset_index()
        chunks.columns = ['chunk_id', 'end_dist', 'start_ele', 'end_ele']

        chunks['dist_delta'] = chunks['end_dist'].diff().fillna(sample_dist)
        chunks['ele_delta'] = chunks['end_ele'] - chunks['start_ele']
        chunks['grade'] = chunks['ele_delta'] / chunks['dist_delta']

        segments = []
        current_type = None
        seg_start_dist = 0
        seg_start_ele = chunks.iloc[0]['start_ele']
        grade_threshold = 0.05

        for i, row in chunks.iterrows():
            if row['grade'] > grade_threshold:
                chunk_type = "Climb"
            elif row['grade'] < -grade_threshold:
                chunk_type = "Descent"
            else:
                chunk_type = "Flat"

            if current_type is None:
                current_type = chunk_type

            if chunk_type != current_type:
                segments.append({
                    "type": current_type,
                    "start_dist": seg_start_dist,
                    "end_dist": row['end_dist'],
                    "start_ele": seg_start_ele,
                    "end_ele": row['end_ele']
                })
                current_type = chunk_type
                seg_start_dist = row['end_dist']
                seg_start_ele = row['start_ele']

        if len(chunks) > 0:
            last_row = chunks.iloc[-1]
            segments.append({
                "type": current_type,
                "start_dist": seg_start_dist,
                "end_dist": last_row['end_dist'],
                "start_ele": seg_start_ele,
                "end_ele": last_row['end_ele']
            })

        # Optimized segment merging - pre-calculate distances
        min_dist_threshold = 800
        while len(segments) > 1:
            # Calculate distances once per iteration
            dists = [(i, s['end_dist'] - s['start_dist']) for i, s in enumerate(segments)]
            shortest_idx, min_dist = min(dists, key=lambda x: x[1])

            if min_dist >= min_dist_threshold:
                break

            left_idx = shortest_idx - 1 if shortest_idx > 0 else None
            right_idx = shortest_idx + 1 if shortest_idx < len(segments) - 1 else None

            if left_idx is not None and right_idx is not None:
                left_dist = segments[left_idx]['end_dist'] - segments[left_idx]['start_dist']
                right_dist = segments[right_idx]['end_dist'] - segments[right_idx]['start_dist']
                neighbor_idx = left_idx if left_dist < right_dist else right_idx
            elif left_idx is not None:
                neighbor_idx = left_idx
            else:
                neighbor_idx = right_idx

            n_seg = segments[neighbor_idx]
            s_seg = segments[shortest_idx]

            new_start_dist = min(n_seg['start_dist'], s_seg['start_dist'])
            new_end_dist = max(n_seg['end_dist'], s_seg['end_dist'])
            new_start_ele = n_seg['start_ele'] if new_start_dist == n_seg['start_dist'] else s_seg['start_ele']
            new_end_ele = n_seg['end_ele'] if new_end_dist == n_seg['end_dist'] else s_seg['end_ele']

            merged_seg = {
                'start_dist': new_start_dist,
                'end_dist': new_end_dist,
                'start_ele': new_start_ele,
                'end_ele': new_end_ele
            }

            min_i = min(shortest_idx, neighbor_idx)
            max_i = max(shortest_idx, neighbor_idx)
            segments.pop(max_i)
            segments.pop(min_i)
            segments.insert(min_i, merged_seg)

        # Final merging pass
        final_merged = []
        final_threshold = 0.05

        for seg in segments:
            dist_m = seg['end_dist'] - seg['start_dist']
            if dist_m == 0:
                continue

            net_ele = seg['end_ele'] - seg['start_ele']
            grade = net_ele / dist_m

            if grade > final_threshold:
                seg['type'] = "Climb"
            elif grade < -final_threshold:
                seg['type'] = "Descent"
            else:
                seg['type'] = "Flat"

            if not final_merged:
                final_merged.append(seg)
            else:
                if final_merged[-1]['type'] == seg['type']:
                    final_merged[-1]['end_dist'] = seg['end_dist']
                    final_merged[-1]['end_ele'] = seg['end_ele']
                else:
                    final_merged.append(seg)

        final_output = []
        for seg in final_merged:
            dist_m = seg['end_dist'] - seg['start_dist']
            if dist_m == 0:
                continue
            dist_km = dist_m / 1000
            net_ele = seg['end_ele'] - seg['start_ele']
            avg_grade = (net_ele / dist_m) * 100

            mask = (df['cum_dist'] >= seg['start_dist']) & (df['cum_dist'] <= seg['end_dist'])
            sub_df = df[mask]
            if len(sub_df) > 1:
                seg_gain = sub_df['cum_gain'].iloc[-1] - sub_df['cum_gain'].iloc[0]
                seg_loss = sub_df['cum_loss'].iloc[-1] - sub_df['cum_loss'].iloc[0]
            else:
                seg_gain, seg_loss = 0, 0

            final_output.append({
                "start_dist": seg['start_dist'],
                "end_dist": seg['end_dist'],
                "Segment Name": f"{seg['type']} ({int(seg['start_dist']/1000)}k)",
                "Dist (km)": round(dist_km, 2),
                "Net Elev (m)": int(net_ele),
                "Gain (m)": int(seg_gain),
                "Loss (m)": int(seg_loss),
                "Grade (%)": round(avg_grade, 1)
            })
        return pd.DataFrame(final_output)

    def _get_segment_times(self, segments_df, history_dfs, scenario="Average", tech_multiplier=1.0):
        """Get estimated times for segments with optimized search and diagnostics."""
        times = []
        debug_stats = {"fallback_all": 0, "generic": 0, "segment_details": []}

        # Pre-cache search indices for each history dataframe with adaptive points
        search_caches = {}
        for name, h_df in history_dfs.items():
            if h_df is not None:
                # Adaptive search points based on route length
                route_km = h_df['cum_dist'].max() / 1000
                num_points = int(np.clip(route_km / 2, self.min_search_points, self.max_search_points))
                # Use linspace instead of arange to avoid huge arrays
                search_dists = np.linspace(h_df['cum_dist'].min(), h_df['cum_dist'].max(), num_points)
                search_caches[name] = np.searchsorted(h_df['cum_dist'].values, search_dists)

        for _, row in segments_df.iterrows():
            target_dist = row['Dist (km)'] * 1000

            is_descent = row['Net Elev (m)'] < 0
            search_col = 'cum_loss' if is_descent else 'cum_gain'
            target_ele_abs = row['Loss (m)'] if is_descent else row['Gain (m)']

            # Auto-optimize tolerances (5-20%) to maximize matches
            best_matches = []
            best_tol_dist = 0.05
            best_tol_ele = 0.05
            tol_options = [0.05, 0.10, 0.15, 0.20]

            for tol_dist in tol_options:
                for tol_ele in tol_options:
                    matches = []

                    for name, h_df in history_dfs.items():
                        if h_df is None or h_df['cum_dist'].max() < target_dist or h_df[search_col].max() < target_ele_abs:
                            continue

                        starts = search_caches.get(name, [])

                        # Cache array values for faster access
                        cum_dist_vals = h_df['cum_dist'].values
                        search_col_vals = h_df[search_col].values
                        time_vals = h_df['time'].values

                        for s_idx in starts:
                            if s_idx >= len(h_df):
                                break
                            s_dist = cum_dist_vals[s_idx]
                            s_ele = search_col_vals[s_idx]
                            s_time = time_vals[s_idx]

                            e_idx = np.searchsorted(cum_dist_vals, s_dist + target_dist)
                            if e_idx >= len(h_df):
                                continue

                            act_dist = cum_dist_vals[e_idx] - s_dist
                            if abs(act_dist - target_dist) > (target_dist * tol_dist):
                                continue

                            act_ele = search_col_vals[e_idx] - s_ele
                            if (target_ele_abs * (1 - tol_ele)) <= act_ele <= (target_ele_abs * (1 + tol_ele)):
                                time_diff = time_vals[e_idx] - s_time
                                elapsed = float(time_diff / np.timedelta64(1, 's'))
                                matches.append(elapsed)
                                break

                    if len(matches) > len(best_matches):
                        best_matches = matches.copy()
                        best_tol_dist = tol_dist
                        best_tol_ele = tol_ele

            matches = best_matches

            seg_name = row['Segment Name']
            seg_dist = row['Dist (km)']
            seg_ele = row['Gain (m)'] if row['Net Elev (m)'] >= 0 else row['Loss (m)']

            if matches:
                debug_stats["fallback_all"] += 1
                match_type = "Training Match"
            else:
                debug_stats["generic"] += 1
                match_type = "Generic Estimate"

            debug_stats["segment_details"].append({
                "name": seg_name,
                "dist_km": round(seg_dist, 2),
                "ele_m": int(seg_ele),
                "match_type": match_type,
                "num_matches": len(matches) if matches else 0,
                "tol_dist": int(best_tol_dist * 100),
                "tol_ele": int(best_tol_ele * 100)
            })

            if matches:
                if len(matches) > 1:
                    if scenario == "Fast (Optimistic)":
                        est_time = np.percentile(matches, 25)
                    elif scenario == "Slow (Conservative)":
                        est_time = np.percentile(matches, 75)
                    else:
                        est_time = np.mean(matches)
                else:
                    if scenario == "Fast (Optimistic)":
                        est_time = matches[0] * 0.95
                    elif scenario == "Slow (Conservative)":
                        est_time = matches[0] * 1.05
                    else:
                        est_time = matches[0]
            else:
                base_time = row['Dist (km)'] * 600
                if scenario == "Fast (Optimistic)":
                    est_time = base_time * 0.90
                elif scenario == "Slow (Conservative)":
                    est_time = base_time * 1.10
                else:
                    est_time = base_time

            est_time = est_time * tech_multiplier
            times.append(est_time)

        # Store debug stats in session state for later display
        st.session_state.segment_match_stats = debug_stats
        return times

    def predict_by_terrain(self, segments_df, history_dfs, num_aid_stations=0, time_per_aid_min=0, scenario="Average", tech_multiplier=1.0):
        """Generate terrain-based pacing plan."""
        plan = []
        cum_time = 0
        total_dist, total_gain, total_loss, total_water, total_carbs = 0, 0, 0, 0, 0

        seg_times = self._get_segment_times(segments_df, history_dfs, scenario, tech_multiplier)

        for idx, row in segments_df.iterrows():
            est_time = seg_times[idx]
            cum_time += est_time

            hours = est_time / 3600
            water = hours * self.nutrition['water']
            carbs = hours * self.nutrition['carbs']

            total_dist += row['Dist (km)']
            total_gain += row['Gain (m)']
            total_loss += row['Loss (m)']
            total_water += water
            total_carbs += carbs

            plan.append({
                "Segment": row['Segment Name'],
                "Dist": f"{row['Dist (km)']}km",
                "Grade": f"{row['Grade (%)']}%",
                "Elev": f"+{int(row['Gain (m)'])}m / -{int(row['Loss (m)'])}m",
                "Time": seconds_to_hms(est_time),
                "Arrival": seconds_to_hms(cum_time),
                "Water": f"{int(water)}oz",
                "Carbs": f"{int(carbs)}g"
            })

        total_aid_sec = int(num_aid_stations * time_per_aid_min * 60)
        if total_aid_sec > 0:
            cum_time += total_aid_sec
            plan.append({
                "Segment": f"⛺ Aid Stations ({num_aid_stations})",
                "Dist": "-", "Grade": "-", "Elev": "-",
                "Time": seconds_to_hms(total_aid_sec),
                "Arrival": "-", "Water": "-", "Carbs": "-"
            })

        plan.append({
            "Segment": "🏁 TOTALS",
            "Dist": f"{total_dist:.1f}km", "Grade": "-",
            "Elev": f"+{int(total_gain)}m / -{int(total_loss)}m",
            "Time": seconds_to_hms(cum_time), "Arrival": "FINISH",
            "Water": f"{int(total_water)}oz", "Carbs": f"{int(total_carbs)}g"
        })
        return plan, cum_time

    def predict_by_time(self, race_df, segments_df, history_dfs, interval_hours=1, num_aid_stations=0, time_per_aid_min=0, scenario="Average", tech_multiplier=1.0):
        """Generate time-based pacing plan."""
        # Create a copy to avoid modifying input
        segments_copy = segments_df.copy()
        seg_times = self._get_segment_times(segments_copy, history_dfs, scenario, tech_multiplier)

        df = race_df.copy()
        df['est_pace'] = 0.6

        # Use enumerate to avoid modifying original dataframe
        for idx, (_, row) in enumerate(segments_copy.iterrows()):
            mask = (df['cum_dist'] >= row['start_dist']) & (df['cum_dist'] < row['end_dist'])
            dist_m = row['end_dist'] - row['start_dist']
            if dist_m > 0:
                df.loc[mask, 'est_pace'] = seg_times[idx] / dist_m

        df['time_diff'] = df['dist_diff'] * df['est_pace']
        df['cum_time'] = df['time_diff'].cumsum()

        interval_sec = interval_hours * 3600
        df['time_bucket'] = (df['cum_time'] // interval_sec).astype(int)

        plan = []
        total_time = df['cum_time'].max()
        total_dist, total_gain, total_loss, total_water, total_carbs = 0, 0, 0, 0, 0

        for bucket, group in df.groupby('time_bucket'):
            b_dist_m = group['dist_diff'].sum()
            b_dist_km = b_dist_m / 1000
            b_time_sec = group['time_diff'].sum()

            diffs = group['ele_diff']
            b_gain = diffs[diffs > 0].sum()
            b_loss = abs(diffs[diffs < 0].sum())

            b_hours = b_time_sec / 3600
            water = b_hours * self.nutrition['water']
            carbs = b_hours * self.nutrition['carbs']

            cum_t = group['cum_time'].values[-1]
            cum_d = group['cum_dist'].values[-1] / 1000

            is_partial = b_time_sec < (interval_sec * 0.95)
            seg_name = f"Hour {bucket + 1}" + (" (Finish)" if is_partial else "")

            plan.append({
                "Time Block": seg_name,
                "Distance": f"{b_dist_km:.2f}km",
                "Total Dist": f"{cum_d:.2f}km",
                "Gain / Loss": f"+{int(b_gain)}m / -{int(b_loss)}m",
                "Water": f"{int(water)}oz",
                "Carbs": f"{int(carbs)}g",
                "Clock Time": seconds_to_hms(cum_t)
            })

            total_dist += b_dist_km
            total_gain += b_gain
            total_loss += b_loss
            total_water += water
            total_carbs += carbs

        total_aid_sec = int(num_aid_stations * time_per_aid_min * 60)
        if total_aid_sec > 0:
            total_time += total_aid_sec
            plan.append({
                "Time Block": f"⛺ Aid Stations ({num_aid_stations})",
                "Distance": "-", "Total Dist": "-", "Gain / Loss": "-",
                "Water": "-", "Carbs": "-",
                "Clock Time": f"+{int(total_aid_sec // 3600)}h {int((total_aid_sec % 3600) // 60):02}m"
            })

        plan.append({
            "Time Block": "🏁 TOTALS",
            "Distance": f"{total_dist:.2f}km", "Total Dist": "-",
            "Gain / Loss": f"+{int(total_gain)}m / -{int(total_loss)}m",
            "Water": f"{int(total_water)}oz", "Carbs": f"{int(total_carbs)}g",
            "Clock Time": seconds_to_hms(total_time)
        })

        return plan, total_time

    def get_chart(self, race_df, segments_df):
        """Generate elevation profile chart."""
        if race_df is None:
            return None
        plot_df = pd.merge_asof(
            race_df.sort_values('cum_dist'),
            segments_df[['start_dist', 'Segment Name']].sort_values('start_dist'),
            left_on='cum_dist',
            right_on='start_dist',
            direction='backward'
        )
        plot_df['Segment Name'] = plot_df['Segment Name'].ffill().bfill()
        # Vectorized string extraction
        plot_df['Type'] = plot_df['Segment Name'].str.extract(r'^(\w+)', expand=False).fillna('Flat')
        plot_df['Distance (km)'] = plot_df['cum_dist'] / 1000

        color_map = {"Climb": "#FF4B4B", "Descent": "#00CC96", "Flat": "#7F7F7F"}
        fig = px.area(plot_df, x='Distance (km)', y='ele', color='Type', color_discrete_map=color_map, template="plotly_dark")
        fig.update_layout(showlegend=True, legend_title_text=None, height=350, margin=dict(l=0, r=0, t=30, b=0))
        return fig

    def get_map(self, race_df):
        """Generate course map."""
        if race_df is None or 'lat' not in race_df.columns:
            return None

        fig = px.line_map(
            race_df.iloc[::5],
            lat="lat",
            lon="lon",
            zoom=11,
            height=350,
            map_style="open-street-map"
        )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        fig.update_traces(line=dict(color='#FF4B4B', width=4))
        return fig


# --- APP UI ---
st.title("Race Planner - Optimized")
logic = RaceLogic()

with st.sidebar:
    st.header("Configuration")
    race_file = st.file_uploader("Upload Race GPX", type=['gpx'])
    hist_files = st.file_uploader("Upload History GPX", type=['gpx'], accept_multiple_files=True)

    st.divider()
    st.subheader("Trail Conditions")
    trail_type = st.selectbox(
        "Expected Technicality:",
        ["Smooth & Fast (Paved/Crushed Dirt)", "Standard Trail (Roots & Rocks)", "Highly Technical (Scrambling/Mud)"],
        index=1
    )

    tech_multiplier = 1.0
    if "Smooth" in trail_type:
        tech_multiplier = 0.90
    elif "Technical" in trail_type:
        tech_multiplier = 1.20

    st.divider()
    st.subheader("Pacing Scenario")
    pace_scenario = st.radio(
        "Select your race day goal:",
        ["Fast (Optimistic)", "Average", "Slow (Conservative)"],
        index=1
    )

    st.divider()
    st.subheader("Nutrition")
    water = st.number_input("Water (oz/hr)", min_value=0, max_value=100, value=20, step=2)
    carbs = st.number_input("Carbs (g/hr)", min_value=0, max_value=200, value=90, step=5)

    logic.nutrition['water'] = water
    logic.nutrition['carbs'] = carbs

    st.divider()
    st.subheader("Aid Stations")
    num_aid_stations = st.number_input("Number of Aid Stations", min_value=0, max_value=50, value=6, step=1)
    time_per_aid = st.number_input("Est. Time per Station (min)", min_value=0.0, max_value=60.0, value=5.0, step=0.5)

    st.divider()
    st.subheader("Matching Tolerances (Advanced)")
    with st.expander("⚙️ Auto-optimized tolerances (5-20%)"):
        st.write("The app automatically selects the best distance and elevation tolerances (up to 20%) for each segment to maximize training matches. No manual adjustment needed.")

    st.divider()
    view_mode = st.radio("Generate Strategy By:", ["Time (1-Hour Blocks)", "Terrain (Climbs & Descents)"])

if race_file:
    race_df = logic.parse_gpx(race_file)
    if race_df is not None:
        segments_df = logic.auto_segment(race_df)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(logic.get_chart(race_df, segments_df), width='stretch')
        with col2: 
            st.plotly_chart(logic.get_map(race_df), width='stretch')
        history_dfs = {}
        if hist_files:
            for f in hist_files:
                h_df = logic.parse_gpx(f)
                if h_df is not None:
                    history_dfs[f.name] = h_df
            if history_dfs:
                st.success(f"✅ Loaded {len(history_dfs)} history files")
            else:
                st.warning("⚠️ No valid history files loaded")
        else:
            st.info("💡 Upload training history files for better pacing predictions")

        st.divider()
        st.subheader(f"Race Strategy Plan: {pace_scenario}")

        if "Time" in view_mode:
            plan, total_time = logic.predict_by_time(
                race_df, segments_df, history_dfs, interval_hours=1,
                num_aid_stations=num_aid_stations, time_per_aid_min=time_per_aid,
                scenario=pace_scenario, tech_multiplier=tech_multiplier
            )
        else:
            plan, total_time = logic.predict_by_terrain(
                segments_df, history_dfs,
                num_aid_stations=num_aid_stations, time_per_aid_min=time_per_aid,
                scenario=pace_scenario, tech_multiplier=tech_multiplier
            )

        # Show segment matching diagnostics
        if "segment_match_stats" in st.session_state:
            stats = st.session_state.segment_match_stats
            with st.expander("📊 Segment Matching Info"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Matched", stats["fallback_all"])
                col2.metric("Generic Estimate", stats["generic"])

                st.write(f"**Tolerances used:** Auto-selected per segment (5-20% range)")

                # Calculate average tolerances
                if stats["segment_details"]:
                    avg_tol_dist = sum(d["tol_dist"] for d in stats["segment_details"]) / len(stats["segment_details"])
                    avg_tol_ele = sum(d["tol_ele"] for d in stats["segment_details"]) / len(stats["segment_details"])
                    st.write(f"**Average tolerances:** Distance ±{avg_tol_dist:.1f}%, Elevation ±{avg_tol_ele:.1f}%")

                if stats["generic"] > 0:
                    st.warning("⚠️ Some segments using generic estimates. Try:")
                    st.write("- Uploading more training data")
                    st.write("- Checking if training data elevation matches race course")

                st.write("**Segment Details:**")
                for detail in stats["segment_details"]:
                    st.write(f"- {detail['name']}: {detail['dist_km']}km, {detail['ele_m']}m | {detail['match_type']} ({detail['num_matches']} matches) | Tol: ±{detail['tol_dist']}% dist, ±{detail['tol_ele']}% elev")

        hours = int(total_time // 3600)
        mins = int((total_time % 3600) // 60)

        c1, c2, c3 = st.columns(3)
        c1.metric("Est. Finish Time", f"{hours}h {mins}m")
        totals = plan[-1]
        c2.metric("Total Water", totals['Water'])
        c3.metric("Total Carbs", totals['Carbs'])

        st.dataframe(pd.DataFrame(plan), width='stretch', hide_index=True)
    else:
        st.error("Could not parse race GPX")
else:
    st.info("📍 Upload your race course (GPX) from the race organizer to begin planning")