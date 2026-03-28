"""
RacePlanMobile_Kivy.py — Android-compatible mobile race planner.
Framework : Kivy  (replaces PySide6, which is desktop-only)
Run locally: python RacePlanMobile_Kivy.py
Build APK  : buildozer android debug
"""

import os
import threading
import traceback

import gpxpy
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

# ─── Palette ──────────────────────────────────────────────────────────────────
BLUE = (0.059, 0.278, 0.380, 1)      # #0F4761
BLUE_LIGHT = (0.231, 0.475, 0.573, 1)
BG = (0.929, 0.957, 0.965, 1)        # #edf4f7
WHITE = (1, 1, 1, 1)
TEXT_DARK = (0.09, 0.09, 0.09, 1)


# ─── Utility functions ────────────────────────────────────────────────────────

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


# ─── Business logic (no PySide6 dependency) ───────────────────────────────────

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

        chunks = (
            df.groupby("chunk_id")
            .agg({"cum_dist": "last", "ele": ["first", "last"]})
            .reset_index()
        )
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
            chunk_type = (
                "Climb" if row["grade"] > grade_threshold
                else "Descent" if row["grade"] < -grade_threshold
                else "Flat"
            )
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
            seg["type"] = (
                "Climb" if grade > final_threshold
                else "Descent" if grade < -final_threshold
                else "Flat"
            )
            if not final_merged or final_merged[-1].get("type") != seg["type"]:
                final_merged.append(seg)
            else:
                final_merged[-1]["end_dist"] = seg["end_dist"]
                final_merged[-1]["end_ele"] = seg["end_ele"]

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
            best_tol_dist = best_tol_ele = 0.05
            tol_options = [0.05, 0.10, 0.15, 0.20]

            for tol_dist in tol_options:
                for tol_ele in tol_options:
                    matches = []
                    for name, h_df in history_dfs.items():
                        if (
                            h_df is None
                            or h_df["cum_dist"].max() < target_dist
                            or h_df[search_col].max() < target_ele_abs
                        ):
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
                            e_idx = np.searchsorted(cum_dist_vals, s_dist + target_dist)
                            if e_idx >= len(h_df):
                                continue
                            act_dist = cum_dist_vals[e_idx] - s_dist
                            if abs(act_dist - target_dist) > (target_dist * tol_dist):
                                continue
                            act_ele = search_col_vals[e_idx] - s_ele
                            if (target_ele_abs * (1 - tol_ele)) <= act_ele <= (target_ele_abs * (1 + tol_ele)):
                                time_diff = time_vals[e_idx] - time_vals[s_idx]
                                elapsed = float(time_diff / np.timedelta64(1, "s"))
                                matches.append(elapsed)
                                break
                    if len(matches) > len(best_matches):
                        best_matches = matches.copy()
                        best_tol_dist, best_tol_ele = tol_dist, tol_ele

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
                    factor = 0.95 if scenario == "Fast (Optimistic)" else 1.05 if scenario == "Slow (Conservative)" else 1.0
                    est_time = matches[0] * factor
            else:
                base_time = row["Dist (km)"] * 600
                factor = 0.90 if scenario == "Fast (Optimistic)" else 1.10 if scenario == "Slow (Conservative)" else 1.0
                est_time = base_time * factor

            times.append(est_time * tech_multiplier)

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
        total_dist = total_gain = total_loss = total_water = total_carbs = 0

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
        total_dist = total_gain = total_loss = total_water = total_carbs = 0

        for bucket, group in df.groupby("time_bucket"):
            b_dist_km = group["dist_diff"].sum() / 1000
            b_time_sec = group["time_diff"].sum()
            diffs = group["ele_diff"]
            b_gain = diffs[diffs > 0].sum()
            b_loss = abs(diffs[diffs < 0].sum())
            hours = b_time_sec / 3600
            water = hours * self.nutrition["water"]
            carbs = hours * self.nutrition["carbs"]
            cum_t = group["cum_time"].values[-1]
            cum_d = group["cum_dist"].values[-1] / 1000
            is_partial = b_time_sec < (interval_sec * 0.95)
            seg_name = f"Hour {bucket + 1}" + (" (Finish)" if is_partial else "")
            plan.append(
                {
                    "Block": seg_name,
                    "Distance": f"{b_dist_km:.2f}km",
                    "Total Dist": f"{cum_d:.2f}km",
                    "Gain/Loss": f"+{int(b_gain)}m/-{int(b_loss)}m",
                    "Water": f"{int(water)}oz",
                    "Carbs": f"{int(carbs)}g",
                    "Clock": seconds_to_hms(cum_t),
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
                    "Block": f"Aid Stations ({num_aid_stations})",
                    "Distance": "-",
                    "Total Dist": "-",
                    "Gain/Loss": "-",
                    "Water": "-",
                    "Carbs": "-",
                    "Clock": f"+{int(total_aid_sec // 3600)}h{int((total_aid_sec % 3600) // 60):02}m",
                }
            )

        plan.append(
            {
                "Block": "TOTALS",
                "Distance": f"{total_dist:.2f}km",
                "Total Dist": "-",
                "Gain/Loss": f"+{int(total_gain)}m/-{int(total_loss)}m",
                "Water": f"{int(total_water)}oz",
                "Carbs": f"{int(total_carbs)}g",
                "Clock": seconds_to_hms(total_time),
            }
        )
        return plan, total_time


# ─── Reusable UI helpers ──────────────────────────────────────────────────────

def _btn(text, h=dp(48), **kwargs):
    return Button(
        text=text,
        size_hint=(1, None),
        height=h,
        background_color=BLUE,
        color=WHITE,
        **kwargs,
    )


def _label(text, h=dp(32), halign="left", **kwargs):
    lbl = Label(
        text=text,
        size_hint=(1, None),
        height=h,
        halign=halign,
        color=TEXT_DARK,
        **kwargs,
    )
    lbl.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0], None)))
    return lbl


def _section_header(text):
    lbl = Label(
        text=f"[b]{text}[/b]",
        markup=True,
        size_hint=(1, None),
        height=dp(36),
        halign="left",
        color=BLUE,
    )
    lbl.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0], None)))
    return lbl


# ─── File-chooser popup ───────────────────────────────────────────────────────

class FilePickerPopup(Popup):
    """
    A modal file-chooser that calls `callback(list_of_paths)` on confirm.
    Works on both desktop (direct file system) and Android (external storage).
    """

    def __init__(self, callback, multi_select=False, **kwargs):
        super().__init__(
            title="Select GPX file(s)",
            size_hint=(0.95, 0.9),
            **kwargs,
        )
        self._callback = callback

        root = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))

        self.chooser = FileChooserListView(
            multiselect=multi_select,
            filters=["*.gpx"],
            path=self._start_path(),
        )
        root.add_widget(self.chooser)

        btn_row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(8))
        btn_row.add_widget(_btn("Cancel", on_press=self.dismiss))
        confirm = _btn("Select")
        confirm.bind(on_press=self._on_confirm)
        btn_row.add_widget(confirm)
        root.add_widget(btn_row)

        self.content = root

    def _start_path(self):
        # On Android the external storage is under /sdcard
        if os.path.exists("/sdcard"):
            return "/sdcard"
        return os.path.expanduser("~")

    def _on_confirm(self, *_args):
        selection = self.chooser.selection
        if selection:
            self.dismiss()
            self._callback(selection)


# ─── Plan card widget ─────────────────────────────────────────────────────────

class PlanCard(BoxLayout):
    """Displays one row (segment or time block) of the race plan."""

    def __init__(self, row_data, **kwargs):
        super().__init__(
            orientation="vertical",
            size_hint_y=None,
            padding=(dp(10), dp(6)),
            spacing=dp(2),
            **kwargs,
        )
        is_total = "TOTAL" in str(list(row_data.values())[0]).upper() or "TOTAL" in str(list(row_data.keys())[0]).upper()

        for key, value in row_data.items():
            row = BoxLayout(size_hint_y=None, height=dp(26))
            key_lbl = Label(
                text=f"[b]{key}[/b]" if not is_total else f"[b]{key}[/b]",
                markup=True,
                size_hint_x=0.4,
                halign="right",
                color=(0.3, 0.3, 0.3, 1) if not is_total else BLUE,
            )
            key_lbl.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0], None)))
            val_lbl = Label(
                text=str(value),
                size_hint_x=0.6,
                halign="left",
                color=TEXT_DARK if not is_total else BLUE,
            )
            val_lbl.bind(size=lambda inst, val: setattr(inst, "text_size", (val[0], None)))
            row.add_widget(key_lbl)
            row.add_widget(val_lbl)
            self.add_widget(row)

        self.height = dp(26) * len(row_data) + dp(16)

        # Separator line
        sep = Widget(size_hint_y=None, height=dp(1))
        self.add_widget(sep)


# ─── Setup tab ────────────────────────────────────────────────────────────────

class SetupTab(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(10), padding=dp(12), **kwargs)
        self._app = app

        scroll = ScrollView(size_hint=(1, 1))
        inner = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10), padding=(0, 0, 0, dp(12)))
        inner.bind(minimum_height=inner.setter("height"))

        # ── File section ──────────────────────────────────────────
        inner.add_widget(_section_header("Files"))

        choose_race = _btn("Select race GPX")
        choose_race.bind(on_press=self._pick_race)
        inner.add_widget(choose_race)

        self.race_label = _label("Race: none", color=(0.4, 0.4, 0.4, 1))
        inner.add_widget(self.race_label)

        add_history = _btn("Add training GPX files")
        add_history.bind(on_press=self._pick_history)
        inner.add_widget(add_history)

        clear_history = _btn("Clear training files")
        clear_history.bind(on_press=self._clear_history)
        inner.add_widget(clear_history)

        self.history_label = _label("Training files: 0", color=(0.4, 0.4, 0.4, 1))
        inner.add_widget(self.history_label)

        # ── Options section ───────────────────────────────────────
        inner.add_widget(_section_header("Plan options"))

        form = GridLayout(cols=2, size_hint_y=None, row_default_height=dp(42), spacing=(dp(4), dp(4)))
        form.bind(minimum_height=form.setter("height"))

        self.scenario = Spinner(
            text="Average",
            values=["Average", "Fast (Optimistic)", "Slow (Conservative)"],
            size_hint=(1, None),
            height=dp(42),
        )
        self.terrain = Spinner(
            text="Normal trail",
            values=["Normal trail", "Technical trail", "Very technical trail"],
            size_hint=(1, None),
            height=dp(42),
        )
        self.view_mode = Spinner(
            text="By segment",
            values=["By segment", "By time block"],
            size_hint=(1, None),
            height=dp(42),
        )

        def _lbl_col(text):
            return Label(text=text, halign="right", color=TEXT_DARK, size_hint=(1, 1))

        form.add_widget(_lbl_col("Scenario"))
        form.add_widget(self.scenario)
        form.add_widget(_lbl_col("Terrain"))
        form.add_widget(self.terrain)
        form.add_widget(_lbl_col("View mode"))
        form.add_widget(self.view_mode)
        inner.add_widget(form)

        # ── Numeric inputs ────────────────────────────────────────
        num_grid = GridLayout(cols=2, size_hint_y=None, row_default_height=dp(42), spacing=(dp(4), dp(4)))
        num_grid.bind(minimum_height=num_grid.setter("height"))

        self.water_input = self._num_input(text="20", hint="Water oz/hr")
        self.carbs_input = self._num_input(text="60", hint="Carbs g/hr")
        self.aid_count_input = self._num_input(text="0", hint="Aid stations")
        self.aid_minutes_input = self._num_input(text="0", hint="Min/aid station")
        self.interval_input = self._num_input(text="1", hint="Time block hrs")

        num_grid.add_widget(_lbl_col("Water (oz/hr)"))
        num_grid.add_widget(self.water_input)
        num_grid.add_widget(_lbl_col("Carbs (g/hr)"))
        num_grid.add_widget(self.carbs_input)
        num_grid.add_widget(_lbl_col("Aid stations"))
        num_grid.add_widget(self.aid_count_input)
        num_grid.add_widget(_lbl_col("Min per aid"))
        num_grid.add_widget(self.aid_minutes_input)
        num_grid.add_widget(_lbl_col("Time block (hrs)"))
        num_grid.add_widget(self.interval_input)
        inner.add_widget(num_grid)

        scroll.add_widget(inner)
        self.add_widget(scroll)

        gen_btn = _btn("Generate race plan", h=dp(54))
        gen_btn.bind(on_press=self._generate)
        self.add_widget(gen_btn)

        self.status_label = _label("Load race GPX then generate a plan.", h=dp(40))
        self.add_widget(self.status_label)

    @staticmethod
    def _num_input(text="0", hint=""):
        return TextInput(
            text=text,
            hint_text=hint,
            multiline=False,
            input_filter="float",
            size_hint=(1, None),
            height=dp(42),
        )

    def _pick_race(self, *_args):
        FilePickerPopup(callback=self._on_race_selected).open()

    def _on_race_selected(self, paths):
        path = paths[0]
        self.status_label.text = "Loading race GPX..."
        threading.Thread(target=self._load_race_bg, args=(path,), daemon=True).start()

    def _load_race_bg(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
            race_df = self._app.logic.parse_gpx(content)
            if race_df is None or race_df.empty:
                raise ValueError("Could not parse GPX file")
            segments_df = self._app.logic.auto_segment(race_df.copy())
            if segments_df.empty:
                raise ValueError("Not enough course data to build segments")
            Clock.schedule_once(lambda dt: self._on_race_loaded(path, race_df, segments_df), 0)
        except Exception:
            err = traceback.format_exc()
            Clock.schedule_once(lambda dt: self._on_race_error(err), 0)

    def _on_race_loaded(self, path, race_df, segments_df):
        self._app.race_df = race_df
        self._app.segments_df = segments_df
        name = os.path.basename(path)
        self.race_label.text = f"Race: {name}"
        self.status_label.text = "Race loaded. Add training files or generate a plan."

    def _on_race_error(self, err):
        self._app.race_df = None
        self._app.segments_df = pd.DataFrame()
        self.race_label.text = "Race: none"
        self.status_label.text = "Error loading race GPX."
        _show_alert("GPX Error", str(err)[:400])

    def _pick_history(self, *_args):
        FilePickerPopup(callback=self._on_history_selected, multi_select=True).open()

    def _on_history_selected(self, paths):
        self.status_label.text = f"Loading {len(paths)} training file(s)..."
        threading.Thread(target=self._load_history_bg, args=(paths,), daemon=True).start()

    def _load_history_bg(self, paths):
        added = 0
        for path in paths:
            if path in self._app.history_dfs:
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                parsed = self._app.logic.parse_gpx(content)
                if parsed is not None and not parsed.empty:
                    self._app.history_dfs[path] = parsed
                    added += 1
            except Exception:
                continue
        Clock.schedule_once(lambda dt: self._on_history_loaded(added), 0)

    def _on_history_loaded(self, added):
        count = len(self._app.history_dfs)
        self.history_label.text = f"Training files: {count}"
        self.status_label.text = (
            f"Added {added} training file(s)." if added else "No valid training files were added."
        )

    def _clear_history(self, *_args):
        self._app.history_dfs = {}
        self.history_label.text = "Training files: 0"
        self.status_label.text = "Training files cleared."

    def _generate(self, *_args):
        if self._app.race_df is None or self._app.segments_df.empty:
            self.status_label.text = "Select a race GPX first."
            return

        try:
            water = int(float(self.water_input.text or "20"))
            carbs = int(float(self.carbs_input.text or "60"))
            num_aid = int(float(self.aid_count_input.text or "0"))
            aid_min = float(self.aid_minutes_input.text or "0")
            interval = int(float(self.interval_input.text or "1"))
        except ValueError:
            self.status_label.text = "Invalid numeric input — check fields."
            return

        self._app.logic.nutrition["water"] = water
        self._app.logic.nutrition["carbs"] = carbs

        scenario = self.scenario.text
        terrain = self.terrain.text
        tech_multiplier = {"Technical trail": 1.10, "Very technical trail": 1.20}.get(terrain, 1.0)
        view_mode = self.view_mode.text
        self.status_label.text = "Generating plan..."

        threading.Thread(
            target=self._generate_bg,
            args=(scenario, tech_multiplier, num_aid, aid_min, view_mode, interval),
            daemon=True,
        ).start()

    def _generate_bg(self, scenario, tech_multiplier, num_aid, aid_min, view_mode, interval):
        try:
            if view_mode == "By segment":
                plan, total_time = self._app.logic.predict_by_terrain(
                    self._app.segments_df,
                    self._app.history_dfs,
                    num_aid_stations=num_aid,
                    time_per_aid_min=aid_min,
                    scenario=scenario,
                    tech_multiplier=tech_multiplier,
                )
            else:
                plan, total_time = self._app.logic.predict_by_time(
                    self._app.race_df,
                    self._app.segments_df,
                    self._app.history_dfs,
                    interval_hours=interval,
                    num_aid_stations=num_aid,
                    time_per_aid_min=aid_min,
                    scenario=scenario,
                    tech_multiplier=tech_multiplier,
                )
            Clock.schedule_once(lambda dt: self._on_plan_ready(plan, total_time), 0)
        except Exception:
            err = traceback.format_exc()
            Clock.schedule_once(lambda dt: self._on_plan_error(err), 0)

    def _on_plan_ready(self, plan, total_time):
        self.status_label.text = "Plan generated."
        self._app.plan_tab.render(plan, total_time)
        self._app.debug_tab.render(self._app.logic.segment_match_stats)
        self._app.tabs.switch_to(self._app.plan_tab_item)

    def _on_plan_error(self, err):
        self.status_label.text = "Plan generation failed. See Debug tab."
        self._app.debug_tab.render_error(err)


# ─── Plan tab ─────────────────────────────────────────────────────────────────

class PlanTab(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(8), padding=dp(8), **kwargs)

        # Summary bar
        summary_box = BoxLayout(size_hint_y=None, height=dp(64), spacing=dp(8))
        self.finish_lbl = self._summary_cell("Finish", "-")
        self.water_lbl = self._summary_cell("Water", "-")
        self.carbs_lbl = self._summary_cell("Carbs", "-")
        summary_box.add_widget(self.finish_lbl)
        summary_box.add_widget(self.water_lbl)
        summary_box.add_widget(self.carbs_lbl)
        self.add_widget(summary_box)

        self.scroll = ScrollView(size_hint=(1, 1))
        self.cards_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(6))
        self.cards_box.bind(minimum_height=self.cards_box.setter("height"))
        self.scroll.add_widget(self.cards_box)
        self.add_widget(self.scroll)

    @staticmethod
    def _summary_cell(label, value):
        box = BoxLayout(orientation="vertical", size_hint=(1, 1))
        box.add_widget(Label(text=label, color=(0.4, 0.4, 0.4, 1), font_size=sp(11)))
        val = Label(text=value, bold=True, color=BLUE, font_size=sp(15))
        box.add_widget(val)
        box._val_label = val
        return box

    def render(self, plan_rows, total_time):
        h = int(total_time // 3600)
        m = int((total_time % 3600) // 60)
        self.finish_lbl._val_label.text = f"{h}h {m:02}m"

        if plan_rows:
            totals = plan_rows[-1]
            self.water_lbl._val_label.text = str(totals.get("Water", "-"))
            self.carbs_lbl._val_label.text = str(totals.get("Carbs", "-"))

        self.cards_box.clear_widgets()
        for row in plan_rows:
            self.cards_box.add_widget(PlanCard(row))


# ─── Debug tab ────────────────────────────────────────────────────────────────

class DebugTab(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=dp(8), padding=dp(8), **kwargs)
        self.text_area = TextInput(
            readonly=True,
            multiline=True,
            font_size=sp(12),
            background_color=(0.98, 1, 1, 1),
        )
        self.add_widget(self.text_area)

    def render(self, stats):
        lines = [
            f"Training matches used : {stats.get('fallback_all', 0)}",
            f"Generic estimates used: {stats.get('generic', 0)}",
            "",
            "Segment details:",
        ]
        for d in stats.get("segment_details", []):
            lines.append(
                f"  {d.get('name', '?')}"
                f"  |  {d.get('match_type', '?')}"
                f"  |  matches: {d.get('num_matches', 0)}"
                f"  |  tol dist: {d.get('tol_dist', 0)}%"
                f"  |  tol elev: {d.get('tol_ele', 0)}%"
            )
        self.text_area.text = "\n".join(lines)

    def render_error(self, err):
        self.text_area.text = err


# ─── Alert helper ─────────────────────────────────────────────────────────────

def _show_alert(title, message):
    content = BoxLayout(orientation="vertical", spacing=dp(8), padding=dp(8))
    content.add_widget(Label(text=message, text_size=(None, None)))
    ok_btn = _btn("OK")
    content.add_widget(ok_btn)
    popup = Popup(title=title, content=content, size_hint=(0.9, 0.5))
    ok_btn.bind(on_press=popup.dismiss)
    popup.open()


# ─── Kivy app ─────────────────────────────────────────────────────────────────

from kivy.utils import get_color_from_hex

try:
    from kivy.metrics import sp
except ImportError:
    def sp(val):
        return val


class RacePlanMobileApp(App):
    def build(self):
        self.title = "Race Planner Mobile"
        self.logic = RaceLogic()
        self.race_df = None
        self.segments_df = pd.DataFrame()
        self.history_dfs = {}

        self.tabs = TabbedPanel(do_default_tab=False, tab_height=dp(50))
        self.tabs.background_color = BG

        # Setup tab
        setup_item = TabbedPanelItem(text="Setup")
        self.setup_tab = SetupTab(app=self)
        setup_item.add_widget(self.setup_tab)
        self.tabs.add_widget(setup_item)

        # Plan tab
        self.plan_tab_item = TabbedPanelItem(text="Plan")
        self.plan_tab = PlanTab()
        self.plan_tab_item.add_widget(self.plan_tab)
        self.tabs.add_widget(self.plan_tab_item)

        # Debug tab
        debug_item = TabbedPanelItem(text="Debug")
        self.debug_tab = DebugTab()
        debug_item.add_widget(self.debug_tab)
        self.tabs.add_widget(debug_item)

        return self.tabs


if __name__ == "__main__":
    RacePlanMobileApp().run()
