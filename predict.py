# prediction.py
import os
import json
from datetime import datetime

import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # ⭐ Cloud 必須
import matplotlib.pyplot as plt

# ======================================================
# 工具：取得目前檔案所在目錄（Cloud 關鍵）
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class HousePricePredictor:
    """
    房價預測與 SHAP 解釋模組（Streamlit Cloud 安全版）
    """

    def __init__(
        self,
        model_path: str = "xgb_house_price_model.pkl",
        feature_path: str = "model_features.pkl",
    ):
        model_path = os.path.join(BASE_DIR, model_path)
        feature_path = os.path.join(BASE_DIR, feature_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型不存在：{model_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"特徵檔不存在：{feature_path}")

        self.model = joblib.load(model_path)
        self.model_features = joblib.load(feature_path)

        self.explainer = shap.TreeExplainer(self.model)

        self.categorical_cols = [
            "district",
            "building_type",
            "main_use",
        ]

    # --------------------------------------------------
    @staticmethod
    def _pretty_name(col: str, value=None) -> str:
        if col.startswith("district_"):
            name = f"行政區：{col.replace('district_', '')}"
        elif col.startswith("building_type_"):
            name = f"建物型態：{col.replace('building_type_', '')}"
        elif col.startswith("main_use_"):
            name = f"主要用途：{col.replace('main_use_', '')}"
        else:
            mapping = {
                "building_age": "屋齡（年）",
                "building_area_sqm": "建物面積（㎡）",
                "main_area": "主建物面積（坪）",
                "balcony_area": "陽台面積（坪）",
                "floor": "樓層",
                "total_floors": "總樓層",
                "has_parking": "車位",
                "has_elevator": "電梯",
            }
            name = mapping.get(col, col)

        return f"{name} = {value}" if value is not None else name

    # --------------------------------------------------
    def _preprocess(self, case_dict: dict) -> pd.DataFrame:
        df = pd.DataFrame([case_dict])

        df = pd.get_dummies(
            df,
            columns=self.categorical_cols,
            drop_first=False,
        )

        missing = set(self.model_features) - set(df.columns)
        if missing:
            df = pd.concat(
                [df, pd.DataFrame(0, index=df.index, columns=list(missing))],
                axis=1,
            )

        return df[self.model_features]

    # --------------------------------------------------
    def predict(self, case_dict: dict) -> float:
        X = self._preprocess(case_dict)
        return float(self.model.predict(X)[0])

    # --------------------------------------------------
    def shap_values(self, case_dict: dict):
        X = self._preprocess(case_dict)
        sv = self.explainer.shap_values(X)
        return sv, X

    # --------------------------------------------------
    def generate_chinese_explanation(self, case_dict: dict, top_n: int = 8) -> str:
        sv, X = self.shap_values(case_dict)

        values = sv[0]
        base = float(self.explainer.expected_value)
        pred = base + values.sum()

        items = sorted(
            zip(X.columns, values, X.iloc[0]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        lines = [
            f"模型平均單價為 {base:.2f} 萬 / 坪，",
            f"此物件預測單價約為 {pred:.2f} 萬 / 坪。",
            "",
            "主要影響因素：",
        ]

        for col, val, data in items:
            direction = "提高" if val > 0 else "降低"
            lines.append(
                f"- {self._pretty_name(col, data)}，{direction}約 {abs(val):.2f} 萬 / 坪"
            )

        return "\n".join(lines)

    # --------------------------------------------------
    def export_prediction_bundle(
        self,
        case_dict: dict,
        output_root: str = "output",
        top_n: int = 8,
    ) -> str:
        output_root = os.path.join(BASE_DIR, output_root)
        os.makedirs(output_root, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(output_root, f"prediction_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        price = self.predict(case_dict)
        explanation = self.generate_chinese_explanation(case_dict, top_n)

        with open(os.path.join(out_dir, "prediction.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "input": case_dict,
                    "predicted_price_wan_per_ping": round(price, 2),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(os.path.join(out_dir, "explanation.txt"), "w", encoding="utf-8") as f:
            f.write(explanation)

        sv, X = self.shap_values(case_dict)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[0],
                base_values=float(self.explainer.expected_value),
                data=X.iloc[0],
                feature_names=X.columns,
            ),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_waterfall.png"), dpi=150)
        plt.close()

        return out_dir
