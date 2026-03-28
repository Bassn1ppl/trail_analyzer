import io
import os
import sys
import traceback

import gpxpy
import numpy as np
import pandas as pd
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import savgol_filter


class WorkerSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class BackgroundTask(QRunnable):
    def __init__(self, task_name, func, *args, **kwargs):
        super().__init__()
        self.task_name = task_name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(self.task_name, result)
        except Exception:
            self.signals.failed.emit(self.task_name, traceback.format_exc())


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371000


def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02}:{s:02}"


class RaceLogic:
    def __init__(self):
        self.nutrition = {"water": 20, "carbs": 60}
        self.min_search_points = 50
        self.max_search_points = 150
        self.segment_match_stats = {"fallback_all": 0, "generic": 0, "segment_details": []}

    def parse_gpx(self, file_content):
        try:
            gpx = gpxpy.parse(file_content)
            data = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        data.append(
                            {
                                "time": point.time,
                                "lat": point.latitude,
                                "lon": point.longitude,
                                "ele": point.elevation or 0,
                            }
                        )

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

            df["ele_diff"] = df["ele"].diff().fillna(0)
            df["cum_gain"] = df["ele_diff"].clip(lower=0).cumsum()
            df["cum_loss"] = df["ele_diff"].clip(upper=0).abs().cumsum()

            return df
        except Exception:
            return None

    def auto_segment(self, df):
        if df.empty or len(df) < 2:
            return pd.DataFrame()

        sample_dist = 100
        df["chunk_id"] = (df["cum_dist"] // sample_dist).astype(int)

        chunks = df.groupby("chunk_id").agg({"cum_dist": "last", "ele": ["first", "last"]}).reset_index()
        chunks.columns = ["chunk_id", "end_dist", "start_ele", "end_ele"]

        chunks["dist_delta"] = chunks["end_dist"].diff().fillna(sample_dist)
        chunks["ele_delta"] = chunks["end_ele"] - chunks["start_ele"]
        chunks["grade"] = chunks["ele_delta"] / chunks["dist_delta"]

        segments = []
        current_type = None
        seg_start_dist = 0
        seg_start_ele = chunks.iloc[0]["start_ele"]
        grade_threshold = 0.05

        for _, row in chunks.iterrows():
            if row["grade"] > grade_threshold:
                chunk_type = "Climb"
            elif row["grade"] < -grade_threshold:
                chunk_type = "Descent"
            else:
                chunk_type = "Flat"

            if current_type is None:
                current_type = chunk_type

            if chunk_type != current_type:
                segments.append(
                    {
                        "type": current_type,
                        "start_dist": seg_start_dist,
                        "end_dist": row["end_dist"],
                        "start_ele": seg_start_ele,
                        "end_ele": row["end_ele"],
                    }
                )
                current_type = chunk_type
                seg_start_dist = row["end_dist"]
                seg_start_ele = row["start_ele"]

        if len(chunks) > 0:
            last_row = chunks.iloc[-1]
            segments.append(
                {
                    "type": current_type,
                    "start_dist": seg_start_dist,
                    "end_dist": last_row["end_dist"],
                    "start_ele": seg_start_ele,
                    "end_ele": last_row["end_ele"],
                }
            )

        min_dist_threshold = 800
        while len(segments) > 1:
            dists = [(i, s["end_dist"] - s["start_dist"]) for i, s in enumerate(segments)]
            shortest_idx, min_dist = min(dists, key=lambda x: x[1])

            if min_dist >= min_dist_threshold:
                break

            left_idx = shortest_idx - 1 if shortest_idx > 0 else None
            right_idx = shortest_idx + 1 if shortest_idx < len(segments) - 1 else None

            if left_idx is not None and right_idx is not None:
                left_dist = segments[left_idx]["end_dist"] - segments[left_idx]["start_dist"]
                right_dist = segments[right_idx]["end_dist"] - segments[right_idx]["start_dist"]
                neighbor_idx = left_idx if left_dist < right_dist else right_idx
            elif left_idx is not None:
                neighbor_idx = left_idx
            else:
                neighbor_idx = right_idx

            n_seg = segments[neighbor_idx]
            s_seg = segments[shortest_idx]

            new_start_dist = min(n_seg["start_dist"], s_seg["start_dist"])
            new_end_dist = max(n_seg["end_dist"], s_seg["end_dist"])
            new_start_ele = n_seg["start_ele"] if new_start_dist == n_seg["start_dist"] else s_seg["start_ele"]
            new_end_ele = n_seg["end_ele"] if new_end_dist == n_seg["end_dist"] else s_seg["end_ele"]

            merged_seg = {
                "start_dist": new_start_dist,
                "end_dist": new_end_dist,
                "start_ele": new_start_ele,
                "end_ele": new_end_ele,
            }

            min_i = min(shortest_idx, neighbor_idx)
            max_i = max(shortest_idx, neighbor_idx)
            segments.pop(max_i)
            segments.pop(min_i)
            segments.insert(min_i, merged_seg)

        final_merged = []
        final_threshold = 0.05

        for seg in segments:
            dist_m = seg["end_dist"] - seg["start_dist"]
            if dist_m == 0:
                continue

            net_ele = seg["end_ele"] - seg["start_ele"]
            grade = net_ele / dist_m

            if grade > final_threshold:
                seg["type"] = "Climb"
            elif grade < -final_threshold:
                seg["type"] = "Descent"
            else:
                seg["type"] = "Flat"

            if not final_merged:
                final_merged.append(seg)
            else:
                if final_merged[-1]["type"] == seg["type"]:
                    final_merged[-1]["end_dist"] = seg["end_dist"]
                    final_merged[-1]["end_ele"] = seg["end_ele"]
                else:
                    final_merged.append(seg)

        final_output = []
        for seg in final_merged:
            dist_m = seg["end_dist"] - seg["start_dist"]
            if dist_m == 0:
                continue
            dist_km = dist_m / 1000
            net_ele = seg["end_ele"] - seg["start_ele"]
            avg_grade = (net_ele / dist_m) * 100

            mask = (df["cum_dist"] >= seg["start_dist"]) & (df["cum_dist"] <= seg["end_dist"])
            sub_df = df[mask]
            if len(sub_df) > 1:
                seg_gain = sub_df["cum_gain"].iloc[-1] - sub_df["cum_gain"].iloc[0]
                seg_loss = sub_df["cum_loss"].iloc[-1] - sub_df["cum_loss"].iloc[0]
            else:
                seg_gain, seg_loss = 0, 0

            final_output.append(
                {
                    "start_dist": seg["start_dist"],
                    "end_dist": seg["end_dist"],
                    "Segment Name": f"{seg['type']} ({int(seg['start_dist'] / 1000)}k)",
                    "Dist (km)": round(dist_km, 2),
                    "Net Elev (m)": int(net_ele),
                    "Gain (m)": int(seg_gain),
                    "Loss (m)": int(seg_loss),
                    "Grade (%)": round(avg_grade, 1),
                }
            )
        return pd.DataFrame(final_output)

    def _get_segment_times(self, segments_df, history_dfs, scenario="Average", tech_multiplier=1.0):
        times = []
        debug_stats = {"fallback_all": 0, "generic": 0, "segment_details": []}

        search_caches = {}
        for name, h_df in history_dfs.items():
            if h_df is not None:
                route_km = h_df["cum_dist"].max() / 1000
                num_points = int(np.clip(route_km / 2, self.min_search_points, self.max_search_points))
                search_dists = np.linspace(h_df["cum_dist"].min(), h_df["cum_dist"].max(), num_points)
                search_caches[name] = np.searchsorted(h_df["cum_dist"].values, search_dists)

        for _, row in segments_df.iterrows():
            target_dist = row["Dist (km)"] * 1000

            is_descent = row["Net Elev (m)"] < 0
            search_col = "cum_loss" if is_descent else "cum_gain"
            target_ele_abs = row["Loss (m)"] if is_descent else row["Gain (m)"]

            best_matches = []
            best_tol_dist = 0.05
            best_tol_ele = 0.05
            tol_options = [0.05, 0.10, 0.15, 0.20]

            for tol_dist in tol_options:
                for tol_ele in tol_options:
                    matches = []

                    for name, h_df in history_dfs.items():
                        if h_df is None or h_df["cum_dist"].max() < target_dist or h_df[search_col].max() < target_ele_abs:
                            continue

                        starts = search_caches.get(name, [])

                        cum_dist_vals = h_df["cum_dist"].values
                        search_col_vals = h_df[search_col].values
                        time_vals = h_df["time"].values

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
                                elapsed = float(time_diff / np.timedelta64(1, "s"))
                                matches.append(elapsed)
                                break

                    if len(matches) > len(best_matches):
                        best_matches = matches.copy()
                        best_tol_dist = tol_dist
                        best_tol_ele = tol_ele

            matches = best_matches

            seg_name = row["Segment Name"]
            seg_dist = row["Dist (km)"]
            seg_ele = row["Gain (m)"] if row["Net Elev (m)"] >= 0 else row["Loss (m)"]

            if matches:
                debug_stats["fallback_all"] += 1
                match_type = "Training Match"
            else:
                debug_stats["generic"] += 1
                match_type = "Generic Estimate"

            debug_stats["segment_details"].append(
                {
                    "name": seg_name,
                    "dist_km": round(seg_dist, 2),
                    "ele_m": int(seg_ele),
                    "match_type": match_type,
                    "num_matches": len(matches) if matches else 0,
                    "tol_dist": int(best_tol_dist * 100),
                    "tol_ele": int(best_tol_ele * 100),
                }
            )

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
                base_time = row["Dist (km)"] * 600
                if scenario == "Fast (Optimistic)":
                    est_time = base_time * 0.90
                elif scenario == "Slow (Conservative)":
                    est_time = base_time * 1.10
                else:
                    est_time = base_time

            est_time = est_time * tech_multiplier
            times.append(est_time)

        self.segment_match_stats = debug_stats
        return times

    def predict_by_terrain(
        self,
        segments_df,
        history_dfs,
        num_aid_stations=0,
        time_per_aid_min=0,
        scenario="Average",
        tech_multiplier=1.0,
        seg_times=None,
    ):
        plan = []
        cum_time = 0
        total_dist, total_gain, total_loss, total_water, total_carbs = 0, 0, 0, 0, 0

        if seg_times is None:
            seg_times = self._get_segment_times(segments_df, history_dfs, scenario, tech_multiplier)

        for idx, row in segments_df.iterrows():
            est_time = seg_times[idx]
            cum_time += est_time

            hours = est_time / 3600
            water = hours * self.nutrition["water"]
            carbs = hours * self.nutrition["carbs"]

            total_dist += row["Dist (km)"]
            total_gain += row["Gain (m)"]
            total_loss += row["Loss (m)"]
            total_water += water
            total_carbs += carbs

            plan.append(
                {
                    "Segment": row["Segment Name"],
                    "Dist": f"{row['Dist (km)']}km",
                    "Grade": f"{row['Grade (%)']}%",
                    "Elev": f"+{int(row['Gain (m)'])}m / -{int(row['Loss (m)'])}m",
                    "Time": seconds_to_hms(est_time),
                    "Arrival": seconds_to_hms(cum_time),
                    "Water": f"{int(water)}oz",
                    "Carbs": f"{int(carbs)}g",
                }
            )

        total_aid_sec = int(num_aid_stations * time_per_aid_min * 60)
        if total_aid_sec > 0:
            cum_time += total_aid_sec
            plan.append(
                {
                    "Segment": f"Aid Stations ({num_aid_stations})",
                    "Dist": "-",
                    "Grade": "-",
                    "Elev": "-",
                    "Time": seconds_to_hms(total_aid_sec),
                    "Arrival": "-",
                    "Water": "-",
                    "Carbs": "-",
                }
            )

        plan.append(
            {
                "Segment": "TOTALS",
                "Dist": f"{total_dist:.1f}km",
                "Grade": "-",
                "Elev": f"+{int(total_gain)}m / -{int(total_loss)}m",
                "Time": seconds_to_hms(cum_time),
                "Arrival": "FINISH",
                "Water": f"{int(total_water)}oz",
                "Carbs": f"{int(total_carbs)}g",
            }
        )
        return plan, cum_time

    def predict_by_time(
        self,
        race_df,
        segments_df,
        history_dfs,
        interval_hours=1,
        num_aid_stations=0,
        time_per_aid_min=0,
        scenario="Average",
        tech_multiplier=1.0,
        seg_times=None,
    ):
        segments_copy = segments_df.copy()
        if seg_times is None:
            seg_times = self._get_segment_times(segments_copy, history_dfs, scenario, tech_multiplier)

        df = race_df.copy()
        df["est_pace"] = 0.6

        for idx, (_, row) in enumerate(segments_copy.iterrows()):
            mask = (df["cum_dist"] >= row["start_dist"]) & (df["cum_dist"] < row["end_dist"])
            dist_m = row["end_dist"] - row["start_dist"]
            if dist_m > 0:
                df.loc[mask, "est_pace"] = seg_times[idx] / dist_m

        df["time_diff"] = df["dist_diff"] * df["est_pace"]
        df["cum_time"] = df["time_diff"].cumsum()

        interval_sec = interval_hours * 3600
        df["time_bucket"] = (df["cum_time"] // interval_sec).astype(int)

        plan = []
        total_time = df["cum_time"].max()
        total_dist, total_gain, total_loss, total_water, total_carbs = 0, 0, 0, 0, 0

        for bucket, group in df.groupby("time_bucket"):
            b_dist_m = group["dist_diff"].sum()
            b_dist_km = b_dist_m / 1000
            b_time_sec = group["time_diff"].sum()

            diffs = group["ele_diff"]
            b_gain = diffs[diffs > 0].sum()
            b_loss = abs(diffs[diffs < 0].sum())

            b_hours = b_time_sec / 3600
            water = b_hours * self.nutrition["water"]
            carbs = b_hours * self.nutrition["carbs"]

            cum_t = group["cum_time"].values[-1]
            cum_d = group["cum_dist"].values[-1] / 1000

            is_partial = b_time_sec < (interval_sec * 0.95)
            seg_name = f"Hour {bucket + 1}" + (" (Finish)" if is_partial else "")

            plan.append(
                {
                    "Time Block": seg_name,
                    "Distance": f"{b_dist_km:.2f}km",
                    "Total Dist": f"{cum_d:.2f}km",
                    "Gain / Loss": f"+{int(b_gain)}m / -{int(b_loss)}m",
                    "Water": f"{int(water)}oz",
                    "Carbs": f"{int(carbs)}g",
                    "Clock Time": seconds_to_hms(cum_t),
                }
            )

            total_dist += b_dist_km
            total_gain += b_gain
            total_loss += b_loss
            total_water += water
            total_carbs += carbs

        total_aid_sec = int(num_aid_stations * time_per_aid_min * 60)
        if total_aid_sec > 0:
            total_time += total_aid_sec
            plan.append(
                {
                    "Time Block": f"Aid Stations ({num_aid_stations})",
                    "Distance": "-",
                    "Total Dist": "-",
                    "Gain / Loss": "-",
                    "Water": "-",
                    "Carbs": "-",
                    "Clock Time": f"+{int(total_aid_sec // 3600)}h {int((total_aid_sec % 3600) // 60):02}m",
                }
            )

        plan.append(
            {
                "Time Block": "TOTALS",
                "Distance": f"{total_dist:.2f}km",
                "Total Dist": "-",
                "Gain / Loss": f"+{int(total_gain)}m / -{int(total_loss)}m",
                "Water": f"{int(total_water)}oz",
                "Carbs": f"{int(total_carbs)}g",
                "Clock Time": seconds_to_hms(total_time),
            }
        )

        return plan, total_time


class RacePlannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Race Planner Pro - PySide6")
        self.resize(1320, 820)

        self.race_path = ""
        self.history_paths = []
        self.history_path_set = set()
        self.race_df = None
        self.segments_df = pd.DataFrame()
        self.history_dfs = {}
        self.initial_analysis_done = False
        self.plan_needs_update = False
        self.thread_pool = QThreadPool.globalInstance()
        self.is_busy = False
        self.queued_task = None

        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Title bar ──────────────────────────────────────────────
        title_bar = QWidget()
        title_bar.setFixedHeight(48)
        title_bar.setStyleSheet("background-color: #0F4761;")
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(16, 0, 16, 0)

        self.title_label = QLabel("Race Planner Pro")
        title_font = QFont("Bahnschrift SemiBold", 16)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #ffffff;")
        title_bar_layout.addWidget(self.title_label)
        title_bar_layout.addStretch()

        # Spinner label — shown during file loading / plan generation
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self.spinner_label = QLabel("⠋  Processing…")
        spinner_font = QFont("Bahnschrift SemiBold", 11)
        self.spinner_label.setFont(spinner_font)
        self.spinner_label.setStyleSheet("color: #ffffff;")
        self.spinner_label.setVisible(False)
        title_bar_layout.addWidget(self.spinner_label)
        title_bar_layout.setContentsMargins(16, 0, 16, 0)

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        root_layout.addWidget(title_bar)

        # ── Main content ───────────────────────────────────────────
        content = QWidget()
        main_layout = QHBoxLayout(content)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)

        root_layout.addWidget(content, 1)

        self._interactive_widgets = [
            self.choose_race_btn,
            self.add_history_btn,
            self.add_history_folder_btn,
            self.clear_history_btn,
            self.history_popup_btn,
            self.trail_combo,
            self.pace_combo,
            self.water_spin,
            self.carbs_spin,
            self.num_aid_spin,
            self.time_per_aid_spin,
            self.view_mode_combo,
            self.update_plan_btn,
            self.rerun_btn,
        ]

    def _show_spinner(self):
        self._spinner_idx = 0
        self.spinner_label.setVisible(True)
        self._spinner_timer.start()
        QApplication.processEvents()

    def _hide_spinner(self):
        self._spinner_timer.stop()
        self.spinner_label.setVisible(False)

    def _tick_spinner(self):
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        frame = self._spinner_frames[self._spinner_idx]
        self.spinner_label.setText(f"{frame}  Processing…")

    def _set_busy(self, busy, status_text=None):
        self.is_busy = busy
        for widget in getattr(self, "_interactive_widgets", []):
            widget.setEnabled(not busy)

        if busy:
            self._show_spinner()
            if status_text:
                self.message_label.setText(status_text)
        else:
            self.update_plan_btn.setEnabled(self.plan_needs_update)
            self._hide_spinner()

    def _start_background_task(self, task_name, func, *args, status_text=None):
        if self.is_busy:
            self.queued_task = (task_name, func, args, status_text)
            return False

        self._set_busy(True, status_text=status_text)
        worker = BackgroundTask(task_name, func, *args)
        worker.signals.finished.connect(self._on_task_finished)
        worker.signals.failed.connect(self._on_task_failed)
        self.thread_pool.start(worker)
        return True

    def _run_queued_task_if_any(self):
        if self.is_busy or self.queued_task is None:
            return

        task_name, func, args, status_text = self.queued_task
        self.queued_task = None
        self._start_background_task(task_name, func, *args, status_text=status_text)

    @Slot(str, object)
    def _on_task_finished(self, task_name, payload):
        self._set_busy(False)

        if task_name == "load_race":
            if not payload["ok"]:
                self.race_df = None
                self.segments_df = pd.DataFrame()
                self.message_label.setText(payload["error"])
                return

            self.race_df = payload["race_df"]
            self.segments_df = payload["segments_df"]
            self.initial_analysis_done = False
            self.plan_needs_update = False
            self.update_plan_btn.setEnabled(False)

            self._render_elevation_profile(self.race_df)
            self.finish_value.setText("-")
            self.water_value.setText("-")
            self.carbs_value.setText("-")
            self.asc_time_value.setText("-")
            self.flat_time_value.setText("-")
            self.desc_time_value.setText("-")
            self._render_table([])

            if self.history_dfs:
                self.generate_plan()
            else:
                self.diagnostics.clear()
                self.diagnostics.setPlaceholderText("Upload history GPX files to compute segment matching info.")
                self.message_label.setText("Race loaded. Elevation profile is ready. Add history GPX files next.")

        elif task_name == "parse_history":
            parsed_history = payload["parsed_history"]

            if not parsed_history:
                self.message_label.setText("No new valid GPX history files were added.")
                return

            for file_path, parsed_df in parsed_history.items():
                self.history_paths.append(file_path)
                self.history_path_set.add(file_path)
                self.history_dfs[file_path] = parsed_df

                item = QListWidgetItem(os.path.basename(file_path))
                item.setToolTip(file_path)
                self.history_popup_list.addItem(item)

            self._update_history_button_text()

            if self.race_df is not None and not self.segments_df.empty:
                self.generate_plan()
            else:
                self.message_label.setText(
                    f"Loaded {len(parsed_history)} history file(s). Choose a race GPX to show elevation and matching info."
                )

        elif task_name == "generate_plan":
            plan = payload["plan"]
            total_time = payload["total_time"]
            stats = payload["stats"]
            history_count = payload["history_count"]

            hours = int(total_time // 3600)
            mins = int((total_time % 3600) // 60)
            totals = plan[-1] if plan else {"Water": "-", "Carbs": "-"}

            self.finish_value.setText(f"{hours}h {mins}m")
            self.water_value.setText(totals.get("Water", "-"))
            self.carbs_value.setText(totals.get("Carbs", "-"))
            self.asc_time_value.setText(payload["asc_time"])
            self.flat_time_value.setText(payload["flat_time"])
            self.desc_time_value.setText(payload["desc_time"])

            if history_count:
                self.message_label.setText(f"Loaded {history_count} history files.")
            else:
                self.message_label.setText("No history files loaded. Using generic estimates where needed.")

            self._render_table(plan)
            self._render_diagnostics(stats)
            self.initial_analysis_done = True
            self.plan_needs_update = False
            self.update_plan_btn.setEnabled(False)

        self._run_queued_task_if_any()

    @Slot(str, str)
    def _on_task_failed(self, task_name, error_text):
        self._set_busy(False)
        self.message_label.setText(f"{task_name} failed. See terminal for details.")
        print(error_text)
        self._run_queued_task_if_any()

    @staticmethod
    def _task_load_race_preview(race_path):
        try:
            with open(race_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                race_stream = io.StringIO(file_obj.read())
        except Exception:
            return {"ok": False, "error": f"Could not read race GPX: {race_path}"}

        logic = RaceLogic()
        race_df = logic.parse_gpx(race_stream)
        if race_df is None:
            return {"ok": False, "error": "Could not parse race GPX file."}

        segments_df = logic.auto_segment(race_df)
        if segments_df.empty:
            return {"ok": False, "error": "Could not generate segments from race GPX."}

        return {"ok": True, "race_df": race_df, "segments_df": segments_df}

    @staticmethod
    def _task_parse_history_files(file_paths):
        logic = RaceLogic()
        parsed_history = {}

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    stream = io.StringIO(file_obj.read())
            except Exception:
                continue

            parsed = logic.parse_gpx(stream)
            if parsed is not None:
                parsed_history[file_path] = parsed

        return {"parsed_history": parsed_history}

    @staticmethod
    def _task_generate_plan(
        race_df,
        segments_df,
        history_dfs,
        water,
        carbs,
        trail_type,
        pace_scenario,
        view_mode,
        num_aid_stations,
        time_per_aid,
    ):
        logic = RaceLogic()
        logic.nutrition["water"] = int(water)
        logic.nutrition["carbs"] = int(carbs)

        tech_multiplier = 1.0
        if "Smooth" in trail_type:
            tech_multiplier = 0.90
        elif "Technical" in trail_type:
            tech_multiplier = 1.20

        if "Time" in view_mode:
            seg_times = logic._get_segment_times(segments_df, history_dfs, pace_scenario, tech_multiplier)
            plan, total_time = logic.predict_by_time(
                race_df,
                segments_df,
                history_dfs,
                interval_hours=1,
                num_aid_stations=int(num_aid_stations),
                time_per_aid_min=float(time_per_aid),
                scenario=pace_scenario,
                tech_multiplier=tech_multiplier,
                seg_times=seg_times,
            )
        else:
            seg_times = logic._get_segment_times(segments_df, history_dfs, pace_scenario, tech_multiplier)
            plan, total_time = logic.predict_by_terrain(
                segments_df,
                history_dfs,
                num_aid_stations=int(num_aid_stations),
                time_per_aid_min=float(time_per_aid),
                scenario=pace_scenario,
                tech_multiplier=tech_multiplier,
                seg_times=seg_times,
            )

        asc_time_sec = 0.0
        flat_time_sec = 0.0
        desc_time_sec = 0.0

        for idx, row in segments_df.iterrows():
            if idx >= len(seg_times):
                continue
            seg_name = str(row.get("Segment Name", ""))
            seg_time = float(seg_times[idx])

            if "Climb" in seg_name:
                asc_time_sec += seg_time
            elif "Descent" in seg_name:
                desc_time_sec += seg_time
            else:
                flat_time_sec += seg_time

        return {
            "plan": plan,
            "total_time": total_time,
            "stats": logic.segment_match_stats,
            "history_count": len(history_dfs),
            "asc_time": seconds_to_hms(asc_time_sec),
            "flat_time": seconds_to_hms(flat_time_sec),
            "desc_time": seconds_to_hms(desc_time_sec),
        }

    @staticmethod
    def _apply_groupbox_subtitle_style(group_box):
        group_box.setStyleSheet(
            """
            QGroupBox {
                margin-top: 18px;
                font-family: "Bahnschrift SemiBold";
                font-size: 13pt;
                font-weight: 600;
                border: none;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 4px;
                color: #0F4761;
            }
            """
        )

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        panel.setMinimumWidth(320)

        files_group = QGroupBox()
        files_group.setTitle("Files")
        self._apply_groupbox_subtitle_style(files_group)
        files_layout = QVBoxLayout(files_group)

        self.race_file_label = QLabel("No race GPX selected")
        self.race_file_label.setWordWrap(True)
        self.choose_race_btn = QPushButton("Choose Race GPX")
        self.choose_race_btn.clicked.connect(self.choose_race_file)

        self.history_popup_btn = QPushButton("History Files (0)")
        self.history_popup_btn.clicked.connect(self._open_history_popup)

        self.history_popup = QDialog(self)
        self.history_popup.setWindowTitle("History Files")
        self.history_popup.resize(560, 420)
        history_popup_layout = QVBoxLayout(self.history_popup)
        self.history_popup_list = QListWidget()
        self.history_popup_list.setStyleSheet("border: none;")
        history_popup_layout.addWidget(self.history_popup_list)
        close_history_popup_btn = QPushButton("Close")
        close_history_popup_btn.clicked.connect(self.history_popup.close)
        history_popup_layout.addWidget(close_history_popup_btn)

        self.add_history_btn = QPushButton("Add History GPX Files")
        self.add_history_btn.clicked.connect(self.add_history_files)
        self.add_history_folder_btn = QPushButton("Add GPX Folder")
        self.add_history_folder_btn.clicked.connect(self.add_history_folder)
        self.clear_history_btn = QPushButton("Clear History Files")
        self.clear_history_btn.clicked.connect(self.clear_history_files)

        files_layout.addWidget(self.choose_race_btn)
        files_layout.addWidget(self.race_file_label)
        files_layout.addWidget(self.add_history_btn)
        files_layout.addWidget(self.add_history_folder_btn)
        files_layout.addWidget(self.clear_history_btn)
        files_layout.addWidget(self.history_popup_btn)

        options_group = QGroupBox()
        options_group.setTitle("Configuration")
        self._apply_groupbox_subtitle_style(options_group)
        options_form = QFormLayout(options_group)

        self.trail_combo = QComboBox()
        self.trail_combo.addItems(
            [
                "Smooth & Fast (Paved/Crushed Dirt)",
                "Standard Trail (Roots & Rocks)",
                "Highly Technical (Scrambling/Mud)",
            ]
        )
        self.trail_combo.setCurrentIndex(1)

        self.pace_combo = QComboBox()
        self.pace_combo.addItems(["Fast (Optimistic)", "Average", "Slow (Conservative)"])
        self.pace_combo.setCurrentText("Average")

        self.water_spin = QSpinBox()
        self.water_spin.setRange(0, 100)
        self.water_spin.setSingleStep(2)
        self.water_spin.setValue(20)

        self.carbs_spin = QSpinBox()
        self.carbs_spin.setRange(0, 200)
        self.carbs_spin.setSingleStep(5)
        self.carbs_spin.setValue(90)

        self.num_aid_spin = QSpinBox()
        self.num_aid_spin.setRange(0, 50)
        self.num_aid_spin.setValue(6)

        self.time_per_aid_spin = QDoubleSpinBox()
        self.time_per_aid_spin.setRange(0, 60)
        self.time_per_aid_spin.setSingleStep(0.5)
        self.time_per_aid_spin.setValue(5.0)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Time (1-Hour Blocks)", "Terrain (Climbs & Descents)"])
        self.view_mode_combo.setCurrentIndex(0)

        options_form.addRow("Trail Conditions", self.trail_combo)
        options_form.addRow("Pacing Scenario", self.pace_combo)
        options_form.addRow("Water (oz/hr)", self.water_spin)
        options_form.addRow("Carbs (g/hr)", self.carbs_spin)
        options_form.addRow("Aid Stations", self.num_aid_spin)
        options_form.addRow("Time per Aid (min)", self.time_per_aid_spin)
        options_form.addRow("Generate Strategy By", self.view_mode_combo)

        self.update_plan_btn = QPushButton("Update Plan")
        self.update_plan_btn.setEnabled(False)
        self.update_plan_btn.clicked.connect(self.generate_plan)

        self.trail_combo.currentTextChanged.connect(self._on_configuration_changed)
        self.pace_combo.currentTextChanged.connect(self._on_configuration_changed)
        self.water_spin.valueChanged.connect(self._on_configuration_changed)
        self.carbs_spin.valueChanged.connect(self._on_configuration_changed)
        self.num_aid_spin.valueChanged.connect(self._on_configuration_changed)
        self.time_per_aid_spin.valueChanged.connect(self._on_configuration_changed)
        self.view_mode_combo.currentTextChanged.connect(self._on_configuration_changed)

        self.rerun_btn = QPushButton("Rerun App")
        self.rerun_btn.setToolTip("Restart the application to pick up code changes")
        self.rerun_btn.clicked.connect(self._rerun_app)

        layout.addWidget(files_group)
        layout.addWidget(options_group)
        layout.addWidget(self.update_plan_btn)
        layout.addWidget(self.rerun_btn)
        layout.addStretch()

        return panel

    def _rerun_app(self):
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _open_diagnostics_popup(self):
        self.diagnostics_popup.show()
        self.diagnostics_popup.raise_()
        self.diagnostics_popup.activateWindow()

    def _open_history_popup(self):
        self.history_popup.show()
        self.history_popup.raise_()
        self.history_popup.activateWindow()

    def _update_history_button_text(self):
        self.history_popup_btn.setText(f"History Files ({len(self.history_paths)})")

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.message_label = QLabel("Select a race GPX to begin.")
        self.message_label.setWordWrap(True)

        elevation_group = QGroupBox()
        elevation_group.setTitle("Elevation Profile")
        self._apply_groupbox_subtitle_style(elevation_group)
        elevation_layout = QVBoxLayout(elevation_group)
        self.elevation_chart = QChart()
        self.elevation_chart.setTitle("Elevation profile will appear after loading a race GPX.")
        self.elevation_chart.legend().hide()

        self.elevation_chart_view = QChartView(self.elevation_chart)
        self.elevation_chart_view.setRenderHint(QPainter.Antialiasing)
        self.elevation_chart_view.setMinimumHeight(260)
        self.elevation_chart_view.setStyleSheet("border: none;")
        elevation_layout.addWidget(self.elevation_chart_view)

        # Diagnostics backing store + popup launcher
        self.diagnostics = QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.diagnostics.setPlaceholderText("Segment matching diagnostics will appear here.")
        self.diagnostics.setStyleSheet("border: none;")
        self.diagnostics.setVisible(False)

        self.segment_info_btn = QPushButton("Segment Matching Info")
        self.segment_info_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.segment_info_btn.clicked.connect(self._open_diagnostics_popup)

        self.diagnostics_popup = QDialog(self)
        self.diagnostics_popup.setWindowTitle("Segment Matching Info")
        self.diagnostics_popup.resize(900, 520)
        popup_layout = QVBoxLayout(self.diagnostics_popup)
        self.diagnostics_popup_summary = QLabel("Segment matching diagnostics will appear here.")
        self.diagnostics_popup_summary.setWordWrap(True)
        popup_layout.addWidget(self.diagnostics_popup_summary)

        self.diagnostics_popup_table = QTableWidget()
        self.diagnostics_popup_table.setAlternatingRowColors(True)
        self.diagnostics_popup_table.setSortingEnabled(False)
        self.diagnostics_popup_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.diagnostics_popup_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        popup_layout.addWidget(self.diagnostics_popup_table, 1)

        close_popup_btn = QPushButton("Close")
        close_popup_btn.clicked.connect(self.diagnostics_popup.close)
        popup_layout.addWidget(close_popup_btn)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.StyledPanel)
        self.table.setLineWidth(1)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)

        # ── Bottom row: summary column + strategy table ────────────
        bottom_row = QWidget()
        bottom_hbox = QHBoxLayout(bottom_row)
        bottom_hbox.setContentsMargins(0, 0, 0, 0)
        bottom_hbox.setSpacing(8)

        metrics_group = QGroupBox()
        metrics_group.setTitle("Summary")
        self._apply_groupbox_subtitle_style(metrics_group)
        metrics_group.setFixedWidth(130)
        metrics_layout = QGridLayout(metrics_group)
        metrics_layout.setContentsMargins(4, 18, 4, 4)
        metrics_layout.setVerticalSpacing(2)
        metrics_layout.setHorizontalSpacing(0)

        lbl_finish = QLabel("Est. Finish Time")
        lbl_finish.setAlignment(Qt.AlignRight)
        self.finish_value = QLabel("-")
        self.finish_value.setAlignment(Qt.AlignRight)

        lbl_water = QLabel("Total Water")
        lbl_water.setAlignment(Qt.AlignRight)
        self.water_value = QLabel("-")
        self.water_value.setAlignment(Qt.AlignRight)

        lbl_carbs = QLabel("Total Carbs")
        lbl_carbs.setAlignment(Qt.AlignRight)
        self.carbs_value = QLabel("-")
        self.carbs_value.setAlignment(Qt.AlignRight)

        lbl_asc = QLabel("Asc Time")
        lbl_asc.setAlignment(Qt.AlignRight)
        self.asc_time_value = QLabel("-")
        self.asc_time_value.setAlignment(Qt.AlignRight)

        lbl_flat = QLabel("Flat Time")
        lbl_flat.setAlignment(Qt.AlignRight)
        self.flat_time_value = QLabel("-")
        self.flat_time_value.setAlignment(Qt.AlignRight)

        lbl_desc = QLabel("Desc Time")
        lbl_desc.setAlignment(Qt.AlignRight)
        self.desc_time_value = QLabel("-")
        self.desc_time_value.setAlignment(Qt.AlignRight)

        metrics_layout.addWidget(lbl_finish, 0, 0)
        metrics_layout.addWidget(self.finish_value, 1, 0)
        metrics_layout.addWidget(lbl_water, 2, 0)
        metrics_layout.addWidget(self.water_value, 3, 0)
        metrics_layout.addWidget(lbl_carbs, 4, 0)
        metrics_layout.addWidget(self.carbs_value, 5, 0)
        metrics_layout.addWidget(lbl_asc, 6, 0)
        metrics_layout.addWidget(self.asc_time_value, 7, 0)
        metrics_layout.addWidget(lbl_flat, 8, 0)
        metrics_layout.addWidget(self.flat_time_value, 9, 0)
        metrics_layout.addWidget(lbl_desc, 10, 0)
        metrics_layout.addWidget(self.desc_time_value, 11, 0)
        metrics_layout.setRowStretch(12, 1)

        bottom_hbox.addWidget(metrics_group)
        bottom_hbox.addWidget(self.table, 1)

        layout.addWidget(elevation_group, 3)
        layout.addWidget(bottom_row, 5)
        layout.addWidget(self.segment_info_btn)

        return panel

    def choose_race_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Race GPX", "", "GPX Files (*.gpx)")
        if file_path:
            self.race_path = file_path
            self.race_file_label.setText(os.path.basename(file_path))
            self.race_file_label.setToolTip(file_path)
            self.title_label.setText(f"Race Planner Pro  ·  {os.path.basename(file_path)}")
            self._load_race_preview()

    def add_history_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose History GPX Files", "", "GPX Files (*.gpx)")
        if not files:
            return

        self._add_history_paths(files)

    def add_history_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Choose Folder With GPX Files")
        if not folder_path:
            return

        gpx_paths = []
        for root_dir, _, file_names in os.walk(folder_path):
            for file_name in file_names:
                if file_name.lower().endswith(".gpx"):
                    gpx_paths.append(os.path.join(root_dir, file_name))

        if not gpx_paths:
            self.message_label.setText("No GPX files found in selected folder.")
            return

        self._add_history_paths(gpx_paths)

    def _add_history_paths(self, file_paths):
        if not file_paths:
            return
        new_paths = [path for path in file_paths if path not in self.history_path_set]
        if not new_paths:
            self.message_label.setText("No new valid GPX history files were added.")
            return

        self._start_background_task(
            "parse_history",
            self._task_parse_history_files,
            new_paths,
            status_text="Parsing history GPX files...",
        )

    def clear_history_files(self):
        self.history_paths = []
        self.history_path_set = set()
        self.history_dfs = {}
        self.initial_analysis_done = False
        self.plan_needs_update = False
        self.history_popup_list.clear()
        self._update_history_button_text()
        self.diagnostics.clear()
        self.diagnostics.setPlaceholderText("Segment matching diagnostics will appear here.")
        self.update_plan_btn.setEnabled(False)
        if self.race_df is not None:
            self.message_label.setText("Race loaded. Add history GPX files to compute segment matching info.")

    def _load_race_preview(self):
        self._start_background_task(
            "load_race",
            self._task_load_race_preview,
            self.race_path,
            status_text="Loading and analyzing race GPX...",
        )

    def _on_configuration_changed(self, *_):
        if self.initial_analysis_done:
            self.plan_needs_update = True
            self.update_plan_btn.setEnabled(True)

    def generate_plan(self):
        if not self.race_path:
            QMessageBox.information(self, "Missing race GPX", "Choose a race GPX file first.")
            return

        if self.race_df is None or self.segments_df.empty:
            self._load_race_preview()
            return

        self._start_background_task(
            "generate_plan",
            self._task_generate_plan,
            self.race_df,
            self.segments_df,
            self.history_dfs.copy(),
            int(self.water_spin.value()),
            int(self.carbs_spin.value()),
            self.trail_combo.currentText(),
            self.pace_combo.currentText(),
            self.view_mode_combo.currentText(),
            int(self.num_aid_spin.value()),
            float(self.time_per_aid_spin.value()),
            status_text="Generating strategy plan...",
        )

    def _render_elevation_profile(self, race_df):
        if race_df is None or race_df.empty:
            return

        distances_km = (race_df["cum_dist"].to_numpy(dtype=float)) / 1000.0
        elevations_m = race_df["ele"].to_numpy(dtype=float)

        max_points = 1400
        num_points = len(distances_km)
        if num_points > max_points:
            sample_idx = np.linspace(0, num_points - 1, max_points).astype(int)
            distances_km = distances_km[sample_idx]
            elevations_m = elevations_m[sample_idx]

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


        def slope_to_color(slope_pct):
            abs_slope = abs(slope_pct)
            if abs_slope > 20:
                return QColor(186, 33, 33, 150)  # Steep (red)
            if abs_slope >= 10:
                return QColor(235, 96, 37, 145)  # Moderate (orange)
            if abs_slope >= 5:
                return QColor(245, 166, 35, 140)  # Slight (yellow)
            return QColor(82, 179, 88, 135)  # Flat (green)

        chart = QChart()
        data_series = []
        


        chart.setTitle(
            f"Elevation Profile • {distances_km[-1]:.2f} km • "
            f"{elevations_m.min():.0f}m to {elevations_m.max():.0f}m"
        )

        chunk_size = 10
        data_series = []
        
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
            pen.setWidth(2)
            chunk_series.setPen(pen)
            chart.addSeries(chunk_series)
            data_series.append(chunk_series)

        chart.legend().setVisible(False)

        chart.addAxis(x_axis, Qt.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignLeft)

        for series in data_series:
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

        self.elevation_chart = chart
        self.elevation_chart_view.setChart(self.elevation_chart)

    def _render_table(self, plan_rows):
        if not plan_rows:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        headers = list(plan_rows[0].keys())
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(plan_rows))

        for row_idx, row_data in enumerate(plan_rows):
            for col_idx, key in enumerate(headers):
                item = QTableWidgetItem(str(row_data.get(key, "")))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _render_diagnostics(self, stats):
        lines = []
        matched_count = stats.get("fallback_all", 0)
        generic_count = stats.get("generic", 0)
        lines.append(f"Training Matched: {matched_count}")
        lines.append(f"Generic Estimate: {generic_count}")
        lines.append("Tolerances used: Auto-selected per segment (5-20% range)")

        details = stats.get("segment_details", [])
        if details:
            avg_tol_dist = sum(d["tol_dist"] for d in details) / len(details)
            avg_tol_ele = sum(d["tol_ele"] for d in details) / len(details)
            lines.append(f"Average tolerances: Distance ±{avg_tol_dist:.1f}%, Elevation ±{avg_tol_ele:.1f}%")
            lines.append("")
            lines.append("Segment Details:")
            for detail in details:
                lines.append(
                    f"- {detail['name']}: {detail['dist_km']}km, {detail['ele_m']}m | "
                    f"{detail['match_type']} ({detail['num_matches']} matches) | "
                    f"Tol: ±{detail['tol_dist']}% dist, ±{detail['tol_ele']}% elev"
                )

        self.diagnostics.setPlainText("\n".join(lines))
        self.diagnostics_popup_summary.setText(
            f"Training Matched: {matched_count}    |    Generic Estimate: {generic_count}"
        )

        table_headers = ["Segment", "Dist (km)", "Elev (m)", "Training Matches", "Tol Dist", "Tol Elev"]
        self.diagnostics_popup_table.clear()
        self.diagnostics_popup_table.setColumnCount(len(table_headers))
        self.diagnostics_popup_table.setHorizontalHeaderLabels(table_headers)
        self.diagnostics_popup_table.setRowCount(len(details))

        for row_idx, detail in enumerate(details):
            row_values = [
                detail.get("name", ""),
                f"{detail.get('dist_km', 0)}",
                f"{detail.get('ele_m', 0)}",
                f"{detail.get('num_matches', 0)}",
                f"±{detail.get('tol_dist', 0)}%",
                f"±{detail.get('tol_ele', 0)}%",
            ]
            for col_idx, value in enumerate(row_values):
                self.diagnostics_popup_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))


def main():
    app = QApplication(sys.argv)
    window = RacePlannerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
