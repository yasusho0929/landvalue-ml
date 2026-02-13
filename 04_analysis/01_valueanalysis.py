"""土地価格・建物価格の予測GUIアプリ。

既存の学習済み joblib モデルを読み込み、
- 土地予測価格 / 土地単価
- 建物予測価格 / 建物単価
を表示する。

主な仕様:
- PySimpleGUI を使用
- 地区名から lat / lng をCSV参照で自動取得
- 文字カテゴリを参照CSVに基づき数値へ変換
- 数値入力チェック
- 未入力・不正入力はポップアップでエラー表示
- 土地のみ / 建物のみ入力でも予測可能
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import traceback

import joblib
import pandas as pd
import PySimpleGUI as sg


# =========================
# パス設定（必要に応じて調整）
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]

LAND_MODEL_CANDIDATES = [
    ROOT_DIR / "02_LO" / "models" / "rf_landprice.joblib",
]
BUILDING_MODEL_CANDIDATES = [
    ROOT_DIR / "03_LAB" / "models" / "rf_building_value.joblib",
]

LAND_COLUMNS_CANDIDATES = [
    ROOT_DIR / "02_LO" / "models" / "rf_landprice_columns.json",
]
BUILDING_COLUMNS_CANDIDATES = [
    ROOT_DIR / "03_LAB" / "models" / "rf_building_value_columns.json",
]

LAND_TRAINING_CSV_CANDIDATES = [
    ROOT_DIR / "02_LO" / "testLO_processed.csv",
]
BUILDING_TRAINING_CSV_CANDIDATES = [
    ROOT_DIR / "03_LAB" / "testLAB_processed.csv",
]

LATLNG_CSV_CANDIDATES = [
    ROOT_DIR / "appendix" / "GifuIchiLatLng_deduplicated.csv",
]

# 文字カテゴリ→数値変換 参照CSV
DISTRICT_PRICE_CSV = ROOT_DIR / "appendix" / "地区名別_坪単価平均.csv"
ROAD_TYPE_PRICE_CSV = ROOT_DIR / "appendix" / "前面道路種類別_坪単価平均.csv"
LAND_SHAPE_PRICE_CSV = ROOT_DIR / "appendix" / "土地形状別_坪単価平均.csv"
STATION_USERS_CSV = ROOT_DIR / "appendix" / "駅の利用者.csv"


# =========================
# モデル想定カラム（フォールバック）
# =========================
LAND_FEATURE_FALLBACK = [
    "NEAREST_STATION",
    "AREA_SQM",
    "LAND_SHAPE",
    "FRONTAGE",
    "DIRECTION",
    "ROAD_TYPE",
    "ROAD_WIDTH",
    "BUILDING_COVERAGE_RATIO",
    "FLOOR_AREA_RATIO",
    "STATION_DISTANCE_MIN",
    "TRANSACTION_YEAR",
    "lat",
    "lng",
]

BUILDING_FEATURE_FALLBACK = [
    "YEAR",
    "RC",
    "MOKU",
    "TETSU",
    "LIGHT_TETSU",
    "BLK",
    "面積（㎡）",
    "TotalArea",
]

# 前処理スクリプト相当の方位マップ
DIRECTION_MAP = {
    "南": 1.0,
    "南東": 0.8,
    "南西": 0.8,
    "東": 0.6,
    "西": 0.6,
    "北東": 0.4,
    "北西": 0.4,
    "北": 0.2,
}


def first_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model(candidates: list[Path], model_name: str) -> Any | None:
    model_path = first_existing_path(candidates)
    if model_path is None:
        sg.popup_error(f"{model_name}モデルが見つかりません。\n候補: {candidates}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        sg.popup_error(f"{model_name}モデルの読み込みに失敗しました。\n{exc}")
        return None


def load_feature_columns(columns_candidates: list[Path], model: Any | None, fallback: list[str]) -> list[str]:
    columns_path = first_existing_path(columns_candidates)
    if columns_path is not None:
        try:
            with columns_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except Exception:  # noqa: BLE001
            pass

    if model is not None and hasattr(model, "feature_names_in_"):
        try:
            cols = list(model.feature_names_in_)
            if cols:
                return cols
        except Exception:  # noqa: BLE001
            pass

    return fallback


def load_training_df(candidates: list[Path]) -> pd.DataFrame | None:
    csv_path = first_existing_path(candidates)
    if csv_path is None:
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def parse_float(value: Any, field_name: str, errors: list[str], required: bool = True) -> float | None:
    text = "" if value is None else str(value).strip()
    if text == "":
        if required:
            errors.append(f"{field_name} は必須です。")
        return None
    try:
        return float(text)
    except ValueError:
        errors.append(f"{field_name} は数値で入力してください。")
        return None


def build_dtype_map(df: pd.DataFrame | None) -> dict[str, str]:
    if df is None:
        return {}
    mapping: dict[str, str] = {}
    for col in df.columns:
        mapping[col] = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "object"
    return mapping


def load_latlng_master(candidates: list[Path]) -> tuple[dict[str, tuple[float, float]], str | None]:
    csv_path = first_existing_path(candidates)
    if csv_path is None:
        return {}, "緯度経度CSVが見つかりません。"

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        return {}, f"緯度経度CSVの読み込みに失敗: {exc}"

    district_col_candidates = ["大字町丁目名", "地区名", "district", "DISTRICT"]
    lat_col_candidates = ["緯度", "lat", "latitude", "LAT"]
    lng_col_candidates = ["経度", "lng", "longitude", "LNG"]

    district_col = next((c for c in district_col_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_col_candidates if c in df.columns), None)
    lng_col = next((c for c in lng_col_candidates if c in df.columns), None)

    if not district_col or not lat_col or not lng_col:
        return {}, "緯度経度CSVに必要列（地区名/緯度/経度）がありません。"

    master: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        district = str(row[district_col]).strip()
        if district == "":
            continue
        try:
            lat = float(row[lat_col])
            lng = float(row[lng_col])
        except (TypeError, ValueError):
            continue
        master[district] = (lat, lng)

    return master, None


def load_two_col_map(csv_path: Path) -> dict[str, float]:
    """2列CSVを {1列目文字列: 2列目数値} で読み込む。"""
    mapping: dict[str, float] = {}
    if not csv_path.exists():
        return mapping

    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            return mapping

        key_col = df.columns[0]
        val_col = df.columns[1]
        for _, row in df.iterrows():
            key = str(row[key_col]).strip()
            if key == "":
                continue
            try:
                val = float(row[val_col])
            except (TypeError, ValueError):
                continue
            mapping[key] = val
    except Exception:
        return {}

    return mapping


def checkbox_to_int(v: Any) -> int:
    return 1 if bool(v) else 0


def main() -> None:
    sg.theme("SystemDefault")

    land_model = load_model(LAND_MODEL_CANDIDATES, "土地")
    building_model = load_model(BUILDING_MODEL_CANDIDATES, "建物")

    if land_model is None and building_model is None:
        sg.popup_error("土地モデル・建物モデルの両方が読み込めないため終了します。")
        return

    land_columns = load_feature_columns(LAND_COLUMNS_CANDIDATES, land_model, LAND_FEATURE_FALLBACK)
    building_columns = load_feature_columns(BUILDING_COLUMNS_CANDIDATES, building_model, BUILDING_FEATURE_FALLBACK)

    land_train_df = load_training_df(LAND_TRAINING_CSV_CANDIDATES)
    building_train_df = load_training_df(BUILDING_TRAINING_CSV_CANDIDATES)
    land_dtype_map = build_dtype_map(land_train_df)
    building_dtype_map = build_dtype_map(building_train_df)

    latlng_master, latlng_error = load_latlng_master(LATLNG_CSV_CANDIDATES)
    if latlng_error:
        sg.popup_error(latlng_error)

    # 参照CSVを使った数値変換マップ
    district_price_map = load_two_col_map(DISTRICT_PRICE_CSV)
    station_users_map = load_two_col_map(STATION_USERS_CSV)
    road_type_map = load_two_col_map(ROAD_TYPE_PRICE_CSV)
    land_shape_map = load_two_col_map(LAND_SHAPE_PRICE_CSV)

    # プルダウン候補（ユーザーが選びやすいよう文字のまま見せる）
    district_values = sorted(set(latlng_master.keys()) | set(district_price_map.keys()))
    land_shape_values = sorted(land_shape_map.keys())
    road_type_values = sorted(road_type_map.keys())
    direction_values = list(DIRECTION_MAP.keys())

    land_frame = [
        [sg.Text("地区名"), sg.Combo(district_values, key="L_DISTRICT", size=(25, 1), enable_events=True)],
        [sg.Text("NEAREST_STATION（駅名）"), sg.Input(key="L_NEAREST_STATION", size=(25, 1))],
        [sg.Text("AREA_SQM（土地面積）"), sg.Input(key="L_AREA_SQM", size=(25, 1))],
        [sg.Text("LAND_SHAPE（土地形状）"), sg.Combo(land_shape_values, key="L_LAND_SHAPE", size=(25, 1))],
        [sg.Text("FRONTAGE（前面道路幅）"), sg.Input(key="L_FRONTAGE", size=(25, 1))],
        [sg.Text("DIRECTION（方位）"), sg.Combo(direction_values, key="L_DIRECTION", size=(25, 1))],
        [sg.Text("ROAD_TYPE（前面道路種類）"), sg.Combo(road_type_values, key="L_ROAD_TYPE", size=(25, 1))],
        [sg.Text("ROAD_WIDTH（道路幅）"), sg.Input(key="L_ROAD_WIDTH", size=(25, 1))],
        [sg.Text("BUILDING_COVERAGE_RATIO"), sg.Input(key="L_BCR", size=(25, 1))],
        [sg.Text("FLOOR_AREA_RATIO"), sg.Input(key="L_FAR", size=(25, 1))],
        [sg.Text("STATION_DISTANCE_MIN"), sg.Input(key="L_STATION_MIN", size=(25, 1))],
        [sg.Text("TRANSACTION_YEAR"), sg.Input(key="L_YEAR", size=(25, 1))],
        [sg.Text("lat（自動）"), sg.Input(key="L_LAT", size=(25, 1), disabled=True)],
        [sg.Text("lng（自動）"), sg.Input(key="L_LNG", size=(25, 1), disabled=True)],
    ]

    # 指示に沿って建物はチェックボックス中心へ変更
    building_frame = [
        [sg.Text("YEAR（築年数）"), sg.Input(key="B_YEAR", size=(25, 1))],
        [sg.Checkbox("RC", key="B_RC")],
        [sg.Checkbox("MOKU", key="B_MOKU")],
        [sg.Checkbox("TETSU", key="B_TETSU")],
        [sg.Checkbox("LIGHT_TETSU", key="B_LIGHT_TETSU")],
        [sg.Checkbox("BLK", key="B_BLK")],
        [sg.Text("TotalArea（延床面積）"), sg.Input(key="B_TOTAL_AREA", size=(25, 1))],
    ]

    result_frame = [
        [sg.Text("土地予測価格:"), sg.Text("-", key="OUT_LAND_PRICE")],
        [sg.Text("土地単価:"), sg.Text("-", key="OUT_LAND_UNIT")],
        [sg.Text("建物予測価格:"), sg.Text("-", key="OUT_BUILDING_PRICE")],
        [sg.Text("建物単価:"), sg.Text("-", key="OUT_BUILDING_UNIT")],
    ]

    layout = [
        [sg.Frame("土地入力", land_frame), sg.VSeparator(), sg.Frame("建物入力", building_frame)],
        [sg.Frame("予測結果", result_frame, expand_x=True)],
        [sg.Button("価格表示"), sg.Button("閉じる")],
    ]

    window = sg.Window("土地・建物価格予測アプリ", layout, finalize=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "閉じる"):
            break

        if event == "L_DISTRICT":
            district = str(values["L_DISTRICT"]).strip()
            if district in latlng_master:
                lat, lng = latlng_master[district]
                window["L_LAT"].update(f"{lat:.8f}")
                window["L_LNG"].update(f"{lng:.8f}")
            else:
                window["L_LAT"].update("")
                window["L_LNG"].update("")

        if event == "価格表示":
            try:
                window["OUT_LAND_PRICE"].update("-")
                window["OUT_LAND_UNIT"].update("-")
                window["OUT_BUILDING_PRICE"].update("-")
                window["OUT_BUILDING_UNIT"].update("-")

                land_has_input = any(str(values[k]).strip() for k in [
                    "L_DISTRICT", "L_NEAREST_STATION", "L_AREA_SQM", "L_LAND_SHAPE",
                    "L_FRONTAGE", "L_DIRECTION", "L_ROAD_TYPE", "L_ROAD_WIDTH",
                    "L_BCR", "L_FAR", "L_STATION_MIN", "L_YEAR",
                ])
                building_has_input = any([
                    str(values["B_YEAR"]).strip() != "",
                    values["B_RC"],
                    values["B_MOKU"],
                    values["B_TETSU"],
                    values["B_LIGHT_TETSU"],
                    values["B_BLK"],
                    str(values["B_TOTAL_AREA"]).strip() != "",
                ])

                if not land_has_input and not building_has_input:
                    sg.popup_error("土地または建物の入力を行ってください。")
                    continue

                if land_has_input:
                    if land_model is None:
                        sg.popup_error("土地モデルが読み込めないため、土地予測はできません。")
                    else:
                        errors: list[str] = []

                        district = str(values["L_DISTRICT"]).strip()
                        station_name = str(values["L_NEAREST_STATION"]).strip()
                        land_shape_name = str(values["L_LAND_SHAPE"]).strip()
                        direction_name = str(values["L_DIRECTION"]).strip()
                        road_type_name = str(values["L_ROAD_TYPE"]).strip()

                        if district == "":
                            errors.append("地区名 は必須です。")
                        if station_name == "":
                            errors.append("NEAREST_STATION（駅名） は必須です。")
                        if land_shape_name == "":
                            errors.append("LAND_SHAPE（土地形状） は必須です。")
                        if direction_name == "":
                            errors.append("DIRECTION（方位） は必須です。")
                        if road_type_name == "":
                            errors.append("ROAD_TYPE（前面道路種類） は必須です。")

                        lat = lng = None
                        if district:
                            if district in latlng_master:
                                lat, lng = latlng_master[district]
                                window["L_LAT"].update(f"{lat:.8f}")
                                window["L_LNG"].update(f"{lng:.8f}")
                            else:
                                errors.append(f"地区名『{district}』の緯度経度がCSVに見つかりません。")

                        # 参照CSV / マップに基づき文字→数値変換
                        station_num = station_users_map.get(station_name)
                        if station_num is None and station_name != "":
                            errors.append(f"駅名『{station_name}』が『駅の利用者.csv』に見つかりません。")

                        land_shape_num = land_shape_map.get(land_shape_name)
                        if land_shape_num is None and land_shape_name != "":
                            errors.append(f"土地形状『{land_shape_name}』が『土地形状別_坪単価平均.csv』に見つかりません。")

                        road_type_num = road_type_map.get(road_type_name)
                        if road_type_num is None and road_type_name != "":
                            errors.append(f"前面道路種類『{road_type_name}』が『前面道路種類別_坪単価平均.csv』に見つかりません。")

                        direction_num = DIRECTION_MAP.get(direction_name)
                        if direction_num is None and direction_name != "":
                            errors.append(f"方位『{direction_name}』が定義済みマップに見つかりません。")

                        gui_to_feature = {
                            "NEAREST_STATION": station_num,
                            "AREA_SQM": values["L_AREA_SQM"],
                            "LAND_SHAPE": land_shape_num,
                            "FRONTAGE": values["L_FRONTAGE"],
                            "DIRECTION": direction_num,
                            "ROAD_TYPE": road_type_num,
                            "ROAD_WIDTH": values["L_ROAD_WIDTH"],
                            "BUILDING_COVERAGE_RATIO": values["L_BCR"],
                            "FLOOR_AREA_RATIO": values["L_FAR"],
                            "STATION_DISTANCE_MIN": values["L_STATION_MIN"],
                            "TRANSACTION_YEAR": values["L_YEAR"],
                            "lat": lat,
                            "lng": lng,
                        }

                        land_row: dict[str, Any] = {}
                        for col in land_columns:
                            raw_value = gui_to_feature.get(col, "")

                            if col in ("lat", "lng"):
                                if raw_value is None:
                                    errors.append(f"{col} の自動取得に失敗しました。")
                                    land_row[col] = None
                                else:
                                    land_row[col] = float(raw_value)
                                continue

                            likely_numeric = (
                                land_dtype_map.get(col) == "numeric"
                                or col in {
                                    "NEAREST_STATION", "AREA_SQM", "LAND_SHAPE", "FRONTAGE", "DIRECTION",
                                    "ROAD_TYPE", "ROAD_WIDTH", "BUILDING_COVERAGE_RATIO", "FLOOR_AREA_RATIO",
                                    "STATION_DISTANCE_MIN", "TRANSACTION_YEAR",
                                }
                            )

                            if likely_numeric:
                                land_row[col] = parse_float(raw_value, col, errors, required=True)
                            else:
                                text = "" if raw_value is None else str(raw_value).strip()
                                if text == "":
                                    errors.append(f"{col} は必須です。")
                                land_row[col] = text

                        if errors:
                            sg.popup_error("土地入力エラー:\n" + "\n".join(errors))
                        else:
                            land_df = pd.DataFrame([land_row], columns=land_columns)
                            land_pred = float(land_model.predict(land_df)[0])

                            area_for_unit = float(land_row.get("AREA_SQM", 0) or 0)
                            if area_for_unit <= 0:
                                sg.popup_error("土地単価計算のため、AREA_SQM は0より大きい値を入力してください。")
                            else:
                                land_unit = land_pred / area_for_unit
                                window["OUT_LAND_PRICE"].update(f"{land_pred:,.2f} 円")
                                window["OUT_LAND_UNIT"].update(f"{land_unit:,.2f} 円/㎡")

                if building_has_input:
                    if building_model is None:
                        sg.popup_error("建物モデルが読み込めないため、建物予測はできません。")
                    else:
                        errors = []

                        year = parse_float(values["B_YEAR"], "YEAR（築年数）", errors, required=True)
                        total_area = parse_float(values["B_TOTAL_AREA"], "TotalArea（延床面積）", errors, required=True)

                        # 指示: チェック有なら1
                        rc = checkbox_to_int(values["B_RC"])
                        moku = checkbox_to_int(values["B_MOKU"])
                        tetsu = checkbox_to_int(values["B_TETSU"])
                        light_tetsu = checkbox_to_int(values["B_LIGHT_TETSU"])
                        blk = checkbox_to_int(values["B_BLK"])

                        # モデル列に合わせて行データ作成
                        gui_to_feature = {
                            "YEAR": year,
                            "RC": rc,
                            "MOKU": moku,
                            "TETSU": tetsu,
                            "LIGHT_TETSU": light_tetsu,
                            "BLK": blk,
                            "面積（㎡）": total_area,
                            "TotalArea": total_area,
                        }

                        building_row: dict[str, Any] = {}
                        for col in building_columns:
                            raw_value = gui_to_feature.get(col, "")
                            likely_numeric = (
                                building_dtype_map.get(col) == "numeric"
                                or col in {"YEAR", "RC", "MOKU", "TETSU", "LIGHT_TETSU", "BLK", "面積（㎡）", "TotalArea"}
                            )
                            if likely_numeric:
                                building_row[col] = parse_float(raw_value, col, errors, required=True)
                            else:
                                text = "" if raw_value is None else str(raw_value).strip()
                                if text == "":
                                    errors.append(f"{col} は必須です。")
                                building_row[col] = text

                        if errors:
                            sg.popup_error("建物入力エラー:\n" + "\n".join(errors))
                        else:
                            building_df = pd.DataFrame([building_row], columns=building_columns)
                            building_pred = float(building_model.predict(building_df)[0])

                            # 単価は延床面積(TotalArea)で計算
                            area_for_unit = float(total_area or 0)
                            if area_for_unit <= 0:
                                sg.popup_error("建物単価計算のため、TotalArea（延床面積）は0より大きい値を入力してください。")
                            else:
                                building_unit = building_pred / area_for_unit
                                window["OUT_BUILDING_PRICE"].update(f"{building_pred:,.2f} 円")
                                window["OUT_BUILDING_UNIT"].update(f"{building_unit:,.2f} 円/㎡")

            except Exception as exc:  # noqa: BLE001
                sg.popup_error(
                    "予測処理中に予期しないエラーが発生しました。\n"
                    f"{exc}\n\n"
                    f"{traceback.format_exc()}"
                )

    window.close()


if __name__ == "__main__":
    main()
