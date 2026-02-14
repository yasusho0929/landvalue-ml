"""Gradioで土地価格・建物価格を予測するGUIアプリ。

要件:
- 既存の学習済みjoblibモデルを利用
- 起動時に各種参照CSVを読み込み
- プルダウン候補は参照CSV/学習CSVから動的生成
- カテゴリ値を参照CSVベースで数値化
- 入力バリデーション（未入力/型/範囲）
- 土地のみ/建物のみでも予測可能
- 予測価格と単価を別表示
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import gradio as gr
import joblib
import pandas as pd


# =========================
# パス設定
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]

LAND_MODEL_CANDIDATES = [
    ROOT_DIR / "02_LO" / "models" / "rf_landprice.joblib",
    ROOT_DIR / "02_LO" / "土地モデル.joblib",
]
BUILDING_MODEL_CANDIDATES = [
    ROOT_DIR / "03_LAB" / "models" / "rf_building_value.joblib",
    ROOT_DIR / "03_LAB" / "建物モデル.joblib",
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

RAW_LAND_CSV_CANDIDATES = [
    ROOT_DIR / "02_LO" / "testL.csv",
]

LATLNG_CSV_CANDIDATES = [
    ROOT_DIR / "appendix" / "GifuIchiLatLng_deduplicated.csv",
]

DISTRICT_PRICE_CSV = ROOT_DIR / "appendix" / "地区名別_坪単価平均.csv"
ROAD_TYPE_PRICE_CSV = ROOT_DIR / "appendix" / "前面道路種類別_坪単価平均.csv"
LAND_SHAPE_PRICE_CSV = ROOT_DIR / "appendix" / "土地形状別_坪単価平均.csv"
STATION_USERS_CSV = ROOT_DIR / "appendix" / "駅の利用者.csv"


# 推論時に使うデフォルト特徴量（学習時カラムJSONがない場合の保険）
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
    "TotalArea",
    "面積（㎡）",
]


# 方角の数値化（学習前処理と整合を取るための既知マップ）
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


@dataclass
class AppResources:
    """起動時に読み込むリソースを集約。"""

    land_model: Any | None
    building_model: Any | None
    land_columns: list[str]
    building_columns: list[str]
    land_training_df: pd.DataFrame | None
    building_training_df: pd.DataFrame | None

    district_map: dict[str, float]
    road_type_map: dict[str, float]
    land_shape_map: dict[str, float]
    station_map: dict[str, float]
    latlng_map: dict[str, tuple[float, float]]

    district_options: list[str]
    station_options: list[str]
    land_shape_options: list[str]
    road_type_options: list[str]
    direction_options: list[str]

    startup_messages: list[str]


def first_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model(candidates: list[Path]) -> tuple[Any | None, str | None]:
    p = first_existing_path(candidates)
    if p is None:
        return None, f"モデルが見つかりません: {candidates}"
    try:
        return joblib.load(p), None
    except Exception as exc:  # noqa: BLE001
        return None, f"モデル読込失敗({p}): {exc}"


def load_json_columns(candidates: list[Path], fallback: list[str]) -> list[str]:
    p = first_existing_path(candidates)
    if p and p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return [str(c) for c in data]
        except Exception:
            pass
    return fallback.copy()


def read_csv_if_exists(candidates: list[Path]) -> pd.DataFrame | None:
    p = first_existing_path(candidates)
    if p is None:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def load_two_col_map(csv_path: Path) -> dict[str, float]:
    """2列以上CSVを読み込んで {先頭列文字列: 2列目数値} を作る。"""
    mapping: dict[str, float] = {}
    if not csv_path.exists():
        return mapping

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return mapping

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
        except Exception:
            continue
        mapping[key] = val

    return mapping


def load_latlng_map(candidates: list[Path]) -> tuple[dict[str, tuple[float, float]], str | None]:
    df = read_csv_if_exists(candidates)
    if df is None:
        return {}, "緯度経度CSVを読み込めませんでした。"

    district_col = next((c for c in ["大字町丁目名", "地区名", "district"] if c in df.columns), None)
    lat_col = next((c for c in ["緯度", "lat", "latitude"] if c in df.columns), None)
    lng_col = next((c for c in ["経度", "lng", "longitude"] if c in df.columns), None)

    if not district_col or not lat_col or not lng_col:
        return {}, "緯度経度CSVに必要列(地区名/緯度/経度)がありません。"

    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        key = str(row[district_col]).strip()
        if not key:
            continue
        try:
            out[key] = (float(row[lat_col]), float(row[lng_col]))
        except Exception:
            continue

    return out, None


def safe_median(mapping: dict[str, float], default: float = 0.0) -> float:
    if not mapping:
        return default
    s = pd.Series(list(mapping.values()), dtype="float64")
    return float(s.median())


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


def format_money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"¥{value:,.0f}"


def align_input_columns(df: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    """学習時カラム順に整列。不足列は0で補完。"""
    aligned = df.copy()
    for col in target_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    return aligned.reindex(columns=target_columns, fill_value=0)


def build_resources() -> AppResources:
    startup_messages: list[str] = []

    land_model, land_err = load_model(LAND_MODEL_CANDIDATES)
    building_model, bld_err = load_model(BUILDING_MODEL_CANDIDATES)

    if land_err:
        startup_messages.append(f"[土地モデル] {land_err}")
    if bld_err:
        startup_messages.append(f"[建物モデル] {bld_err}")

    land_columns = load_json_columns(LAND_COLUMNS_CANDIDATES, LAND_FEATURE_FALLBACK)
    building_columns = load_json_columns(BUILDING_COLUMNS_CANDIDATES, BUILDING_FEATURE_FALLBACK)

    land_training_df = read_csv_if_exists(LAND_TRAINING_CSV_CANDIDATES)
    building_training_df = read_csv_if_exists(BUILDING_TRAINING_CSV_CANDIDATES)

    district_map = load_two_col_map(DISTRICT_PRICE_CSV)
    road_type_map = load_two_col_map(ROAD_TYPE_PRICE_CSV)
    land_shape_map = load_two_col_map(LAND_SHAPE_PRICE_CSV)
    station_map = load_two_col_map(STATION_USERS_CSV)
    latlng_map, latlng_err = load_latlng_map(LATLNG_CSV_CANDIDATES)

    if latlng_err:
        startup_messages.append(f"[緯度経度] {latlng_err}")

    # プルダウン候補をCSV起点で動的構築
    district_options = sorted(district_map.keys())
    station_options = sorted(station_map.keys())
    land_shape_options = sorted(land_shape_map.keys())
    road_type_options = sorted(road_type_map.keys())

    # 方角候補は「生データ(testL.csv)」優先、なければ既知マップを使用
    direction_options: list[str] = []
    raw_df = read_csv_if_exists(RAW_LAND_CSV_CANDIDATES)
    if raw_df is not None and "前面道路：方位" in raw_df.columns:
        direction_options = sorted(
            [str(v).strip() for v in raw_df["前面道路：方位"].dropna().unique() if str(v).strip()]
        )
    if not direction_options:
        direction_options = sorted(DIRECTION_MAP.keys())

    return AppResources(
        land_model=land_model,
        building_model=building_model,
        land_columns=land_columns,
        building_columns=building_columns,
        land_training_df=land_training_df,
        building_training_df=building_training_df,
        district_map=district_map,
        road_type_map=road_type_map,
        land_shape_map=land_shape_map,
        station_map=station_map,
        latlng_map=latlng_map,
        district_options=district_options,
        station_options=station_options,
        land_shape_options=land_shape_options,
        road_type_options=road_type_options,
        direction_options=direction_options,
        startup_messages=startup_messages,
    )


def resolve_category_value(
    key: str,
    mapping: dict[str, float],
    warnings: list[str],
    label: str,
    fallback: float,
) -> float:
    """カテゴリ文字列を参照CSVの値へ変換。未知カテゴリはfallbackに置換して警告。"""
    key = (key or "").strip()
    if key in mapping:
        return mapping[key]
    warnings.append(f"{label}: 未知カテゴリ『{key or '(空)'}』のため代替値 {fallback:.3f} を使用")
    return fallback


def predict(
    district_name: str,
    nearest_station: str,
    area_sqm: str,
    land_shape: str,
    frontage: str,
    direction: str,
    road_type: str,
    road_width: str,
    bcr: str,
    far: str,
    station_distance: str,
    transaction_year: str,
    building_year: str,
    rc: bool,
    moku: bool,
    tetsu: bool,
    light_tetsu: bool,
    blk: bool,
    total_area: str,
) -> tuple[str, str, str, str]:
    errors: list[str] = []
    warnings: list[str] = []

    # 入力有無判定（どちら側を推論するか）
    land_inputs = [
        district_name,
        nearest_station,
        area_sqm,
        land_shape,
        frontage,
        direction,
        road_type,
        road_width,
        bcr,
        far,
        station_distance,
        transaction_year,
    ]
    has_land_input = any(str(v).strip() != "" for v in land_inputs)

    building_inputs = [building_year, total_area, rc, moku, tetsu, light_tetsu, blk]
    has_building_input = any((bool(v) if isinstance(v, bool) else str(v).strip() != "") for v in building_inputs)

    land_result = "土地: 入力なし"
    building_result = "建物: 入力なし"
    ratio_result = "比率: 算出不可"
    land_total_value: float | None = None
    building_total_value: float | None = None

    # -----------------
    # 土地予測
    # -----------------
    if has_land_input:
        if RES.land_model is None:
            errors.append("土地モデルが読み込めないため、土地予測は実行できません。")
        else:
            area = parse_float(area_sqm, "AREA_SQM", errors)
            front = parse_float(frontage, "FRONTAGE", errors)
            r_width = parse_float(road_width, "ROAD_WIDTH", errors)
            val_bcr = parse_float(bcr, "BUILDING_COVERAGE_RATIO", errors)
            val_far = parse_float(far, "FLOOR_AREA_RATIO", errors)
            dist = parse_float(station_distance, "STATION_DISTANCE_MIN", errors)
            year = parse_float(transaction_year, "TRANSACTION_YEAR", errors)

            district_name = (district_name or "").strip()
            nearest_station = (nearest_station or "").strip()
            land_shape = (land_shape or "").strip()
            direction = (direction or "").strip()
            road_type = (road_type or "").strip()

            for field_name, text in [
                ("地区名", district_name),
                ("NEAREST_STATION", nearest_station),
                ("LAND_SHAPE", land_shape),
                ("DIRECTION", direction),
                ("ROAD_TYPE", road_type),
            ]:
                if text == "":
                    errors.append(f"{field_name} は必須です。")

            # 範囲チェック（必要最低限）
            if area is not None and area <= 0:
                errors.append("AREA_SQM は 0 より大きい値を入力してください。")
            if front is not None and front <= 0:
                errors.append("FRONTAGE は 0 より大きい値を入力してください。")
            if r_width is not None and r_width <= 0:
                errors.append("ROAD_WIDTH は 0 より大きい値を入力してください。")
            if val_bcr is not None and not (0 <= val_bcr <= 1000):
                errors.append("BUILDING_COVERAGE_RATIO は 0〜1000 の範囲で入力してください。")
            if val_far is not None and not (0 <= val_far <= 5000):
                errors.append("FLOOR_AREA_RATIO は 0〜5000 の範囲で入力してください。")
            if dist is not None and dist < 0:
                errors.append("STATION_DISTANCE_MIN は 0 以上で入力してください。")

            lat, lng = None, None
            if district_name:
                latlng = RES.latlng_map.get(district_name)
                if latlng is None:
                    warnings.append(f"地区名『{district_name}』の緯度経度が見つからないため 0 を使用")
                    lat, lng = 0.0, 0.0
                else:
                    lat, lng = latlng

            if not errors:
                # 参照CSVからカテゴリ値を数値化
                station_fallback = safe_median(RES.station_map, default=0.0)
                land_shape_fallback = safe_median(RES.land_shape_map, default=0.0)
                road_type_fallback = safe_median(RES.road_type_map, default=0.0)

                station_num = resolve_category_value(
                    nearest_station,
                    RES.station_map,
                    warnings,
                    "NEAREST_STATION",
                    station_fallback,
                )
                land_shape_num = resolve_category_value(
                    land_shape,
                    RES.land_shape_map,
                    warnings,
                    "LAND_SHAPE",
                    land_shape_fallback,
                )
                road_type_num = resolve_category_value(
                    road_type,
                    RES.road_type_map,
                    warnings,
                    "ROAD_TYPE",
                    road_type_fallback,
                )

                # DIRECTIONは既知マップ優先。未知は中央値代替。
                direction_values = list(DIRECTION_MAP.values())
                direction_fallback = float(pd.Series(direction_values).median())
                direction_num = DIRECTION_MAP.get(direction)
                if direction_num is None:
                    warnings.append(f"DIRECTION: 未知カテゴリ『{direction}』のため代替値 {direction_fallback:.3f} を使用")
                    direction_num = direction_fallback

                # 地区名別平均坪単価は、明示特徴量には使わないケースもあるため
                # 「補助情報」として読み込み維持。将来の特徴量追加に備える。
                _district_num = resolve_category_value(
                    district_name,
                    RES.district_map,
                    warnings,
                    "地区名",
                    safe_median(RES.district_map, default=0.0),
                )

                land_row = {
                    "NEAREST_STATION": station_num,
                    "AREA_SQM": area,
                    "LAND_SHAPE": land_shape_num,
                    "FRONTAGE": front,
                    "DIRECTION": direction_num,
                    "ROAD_TYPE": road_type_num,
                    "ROAD_WIDTH": r_width,
                    "BUILDING_COVERAGE_RATIO": val_bcr,
                    "FLOOR_AREA_RATIO": val_far,
                    "STATION_DISTANCE_MIN": dist,
                    "TRANSACTION_YEAR": year,
                    "lat": lat if lat is not None else 0.0,
                    "lng": lng if lng is not None else 0.0,
                }

                X_land = pd.DataFrame([land_row])
                X_land = align_input_columns(X_land, RES.land_columns)

                try:
                    land_pred = float(RES.land_model.predict(X_land)[0])
                    land_total = land_pred * area / 0.3025
                    unit_price = land_total / area if area and area > 0 else None
                    land_total_value = land_total
                    land_result = (
                        f"土地予測価格: {format_money(land_total)}\n"
                        f"土地単価(円/㎡): {format_money(unit_price)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"土地予測に失敗しました: {exc}")

    # -----------------
    # 建物予測
    # -----------------
    if has_building_input:
        if RES.building_model is None:
            errors.append("建物モデルが読み込めないため、建物予測は実行できません。")
        else:
            yr = parse_float(building_year, "YEAR", errors)
            t_area = parse_float(total_area, "TotalArea", errors)

            if yr is not None and not (0 <= yr <= 200):
                errors.append("YEAR(築年数) は 0〜200 の範囲で入力してください。")
            if t_area is not None and t_area <= 0:
                errors.append("TotalArea は 0 より大きい値を入力してください。")

            if not errors:
                bld_row = {
                    "YEAR": yr,
                    "RC": int(bool(rc)),
                    "MOKU": int(bool(moku)),
                    "TETSU": int(bool(tetsu)),
                    "LIGHT_TETSU": int(bool(light_tetsu)),
                    "BLK": int(bool(blk)),
                    "TotalArea": t_area,
                    "面積（㎡）": t_area,
                }

                X_bld = pd.DataFrame([bld_row])
                X_bld = align_input_columns(X_bld, RES.building_columns)

                try:
                    bld_pred = float(RES.building_model.predict(X_bld)[0])
                    bld_unit = bld_pred / t_area if t_area and t_area > 0 else None
                    building_total_value = bld_pred
                    building_result = (
                        f"建物予測価格: {format_money(bld_pred)}\n"
                        f"建物単価(円/㎡): {format_money(bld_unit)}"
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"建物予測に失敗しました: {exc}")

    if land_total_value is not None and building_total_value not in (None, 0):
        ratio = round((land_total_value / building_total_value) * 100, 2)
        ratio_result = f"土地/建物 比率: {ratio:.2f}%"

    if not has_land_input and not has_building_input:
        errors.append("土地または建物の入力を行ってください。")

    message_lines: list[str] = []
    if errors:
        message_lines.append("【エラー】")
        message_lines.extend(f"- {e}" for e in errors)
    if warnings:
        message_lines.append("【警告】")
        message_lines.extend(f"- {w}" for w in warnings)

    if not message_lines:
        message_lines = ["処理が完了しました。"]

    return land_result, building_result, ratio_result, "\n".join(message_lines)


def clear_all() -> tuple[Any, ...]:
    """入力初期化＋出力初期化。"""
    return (
        None,  # district
        None,  # station
        "",  # area
        None,  # shape
        "",  # frontage
        None,  # direction
        None,  # road
        "",  # road width
        "",  # bcr
        "",  # far
        "",  # station distance
        "",  # transaction year
        "",  # building year
        False,  # rc
        False,  # moku
        False,  # tetsu
        False,  # light_tetsu
        False,  # blk
        "",  # total area
        "土地: 入力なし",
        "建物: 入力なし",
        "比率: 算出不可",
        "",
    )


RES = build_resources()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="土地・建物 価格予測") as demo:
        gr.Markdown("## 土地・建物 価格予測（Gradio）")
        if RES.startup_messages:
            gr.Markdown("\n".join(f"- ⚠️ {m}" for m in RES.startup_messages))

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 土地入力")
                district = gr.Dropdown(choices=RES.district_options, label="地区名", allow_custom_value=True)
                station = gr.Dropdown(choices=RES.station_options, label="NEAREST_STATION", allow_custom_value=True)
                area = gr.Textbox(label="土地面積")
                shape = gr.Dropdown(choices=RES.land_shape_options, label="LAND_SHAPE", allow_custom_value=True)
                frontage = gr.Textbox(label="間口")
                direction = gr.Dropdown(choices=RES.direction_options, label="DIRECTION", allow_custom_value=True)
                road_type = gr.Dropdown(choices=RES.road_type_options, label="ROAD_TYPE", allow_custom_value=True)
                road_width = gr.Textbox(label="道路幅")
                bcr = gr.Textbox(label="建蔽率")
                far = gr.Textbox(label="容積率")
                station_distance = gr.Textbox(label="STATION_DISTANCE_MIN")
                tx_year = gr.Textbox(label="成約年")

            with gr.Column():
                gr.Markdown("### 建物入力")
                bld_year = gr.Textbox(label="YEAR（築年数）")
                rc = gr.Checkbox(label="RC")
                moku = gr.Checkbox(label="MOKU")
                tetsu = gr.Checkbox(label="TETSU")
                light_tetsu = gr.Checkbox(label="LIGHT_TETSU")
                blk = gr.Checkbox(label="BLK")
                total_area = gr.Textbox(label="延床面積")

        with gr.Row():
            btn_predict = gr.Button("価格表示", variant="primary")
            btn_clear = gr.Button("クリア")

        with gr.Row():
            land_out = gr.Textbox(label="土地（予測価格・単価）", lines=4)
            bld_out = gr.Textbox(label="建物（予測価格・単価）", lines=4)
            ratio_out = gr.Textbox(label="土地/建物 比率", lines=1)
        msg_out = gr.Textbox(label="エラー / 注意メッセージ", lines=8)

        inputs = [
            district,
            station,
            area,
            shape,
            frontage,
            direction,
            road_type,
            road_width,
            bcr,
            far,
            station_distance,
            tx_year,
            bld_year,
            rc,
            moku,
            tetsu,
            light_tetsu,
            blk,
            total_area,
        ]
        outputs = [land_out, bld_out, ratio_out, msg_out]

        btn_predict.click(fn=predict, inputs=inputs, outputs=outputs)
        btn_clear.click(
            fn=clear_all,
            inputs=[],
            outputs=inputs + outputs,
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
