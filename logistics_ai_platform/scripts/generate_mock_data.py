from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
ORDER_ROWS = 7926
UNIQUE_ORDER_ROWS = 7914
PRODUCT_ROWS = 200
OUTLET_ROWS = 60
RETURN_ROWS = 537
END_DATE = pd.Timestamp("2026-06-15")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
WORKFLOW_DIR = PROJECT_ROOT / "workflows"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
WORKFLOW_RESULT_DIR = PROJECT_ROOT / "outputs" / "workflow_results"

GENERATED_RAW_FILES = [
    "orders.csv",
    "delivery.csv",
    "inventory.csv",
    "products.csv",
    "outlets.csv",
    "returns.csv",
    "customer_complaints.csv",
    "warehouses.csv",
    "delivery_points.csv",
    "README_数据说明.md",
]

GENERATED_REPORT_FILES = [
    PROCESSED_DIR / "data_quality_report.csv",
    WORKFLOW_RESULT_DIR / "data_quality_report.csv",
]

CATEGORIES = ["生鲜粮油", "零食饮料", "日用品", "厨具", "小家电配件", "农产品", "冷链食品"]
FRESH_CATEGORIES = {"生鲜粮油", "农产品", "冷链食品"}
LOW_FREQ_CATEGORIES = {"厨具", "小家电配件"}

PRODUCT_NAME_PARTS = {
    "生鲜粮油": ["东北大米", "花生油", "鸡蛋", "小麦面粉", "玉米糁", "杂粮包"],
    "零食饮料": ["饼干", "果汁", "矿泉水", "坚果", "茶饮料", "山楂糕"],
    "日用品": ["抽纸", "洗衣液", "垃圾袋", "牙膏", "毛巾", "洗洁精"],
    "厨具": ["炒锅", "汤锅", "菜刀", "砧板", "饭盒", "保温壶"],
    "小家电配件": ["净水器滤芯", "电饭煲内胆", "插线板", "风扇电机", "榨汁机刀头", "电水壶底座"],
    "农产品": ["鲜玉米", "红薯", "土豆", "西红柿", "苹果", "蘑菇"],
    "冷链食品": ["冷冻水饺", "冷鲜鸡腿", "酸奶", "速冻馄饨", "冰鲜牛肉", "冷冻汤圆"],
}


def ensure_dirs() -> None:
    for folder in [RAW_DIR, CLEANED_DIR, PROCESSED_DIR, WORKFLOW_DIR, LOG_DIR, WORKFLOW_RESULT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def cleanup_previous_raw_files() -> None:
    for file_name in GENERATED_RAW_FILES:
        file_path = RAW_DIR / file_name
        if file_path.exists():
            file_path.unlink()
    for file_path in GENERATED_REPORT_FILES:
        if file_path.exists():
            file_path.unlink()


def generate_products(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    category_counts = {
        "生鲜粮油": 34,
        "零食饮料": 32,
        "日用品": 34,
        "厨具": 28,
        "小家电配件": 24,
        "农产品": 24,
        "冷链食品": 24,
    }
    category_config = {
        "生鲜粮油": {"price": (6, 90), "margin": (0.12, 0.26), "life": (5, 180), "weight": 4.2, "key_prob": 0.36},
        "零食饮料": {"price": (3, 60), "margin": (0.18, 0.38), "life": (120, 540), "weight": 2.3, "key_prob": 0.18},
        "日用品": {"price": (5, 120), "margin": (0.16, 0.34), "life": (365, 1095), "weight": 1.5, "key_prob": 0.10},
        "厨具": {"price": (18, 360), "margin": (0.20, 0.42), "life": (1095, 3650), "weight": 0.45, "key_prob": 0.04},
        "小家电配件": {"price": (12, 260), "margin": (0.18, 0.40), "life": (1095, 3650), "weight": 0.38, "key_prob": 0.05},
        "农产品": {"price": (4, 68), "margin": (0.10, 0.24), "life": (3, 45), "weight": 3.8, "key_prob": 0.34},
        "冷链食品": {"price": (12, 138), "margin": (0.14, 0.30), "life": (7, 90), "weight": 3.7, "key_prob": 0.38},
    }

    products = []
    profile_rows = []
    product_no = 1
    for category, count in category_counts.items():
        config = category_config[category]
        names = PRODUCT_NAME_PARTS[category]
        for idx in range(count):
            unit_price = round(float(rng.uniform(*config["price"])), 2)
            margin = float(rng.uniform(*config["margin"]))
            cost_price = round(unit_price * (1 - margin), 2)
            is_key_product = int(rng.random() < config["key_prob"])
            product_id = f"P{product_no:04d}"
            products.append(
                {
                    "product_id": product_id,
                    "product_name": f"{names[idx % len(names)]}{idx // len(names) + 1}号",
                    "category": category,
                    "unit_price": unit_price,
                    "cost_price": cost_price,
                    "gross_profit_rate": round((unit_price - cost_price) / unit_price, 4),
                    "shelf_life_days": int(rng.integers(config["life"][0], config["life"][1] + 1)),
                    "is_fresh": int(category in FRESH_CATEGORIES),
                    "is_key_product": is_key_product,
                }
            )
            profile_rows.append(
                {
                    "product_id": product_id,
                    "sales_weight": config["weight"] * (2.2 if is_key_product else 1.0) * float(rng.lognormal(0, 0.28)),
                    "category": category,
                }
            )
            product_no += 1

    return pd.DataFrame(products), pd.DataFrame(profile_rows).set_index("product_id")


def generate_outlets(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_lon, base_lat = 115.48, 35.23
    segments = [
        ("城区常规片区", "城区", 20, ["牡丹街道", "南城街道", "东城街道", "开发区"], base_lon, base_lat, (4, 5), (4, 5), (130, 240), (0.02, 0.07), (0.02, 0.05)),
        ("东北片区", "乡镇", 14, ["小留镇", "黄堽镇", "胡集镇", "高庄镇"], base_lon + 0.24, base_lat + 0.20, (2, 4), (2, 4), (45, 120), (0.16, 0.31), (0.05, 0.09)),
        ("西南片区", "乡镇", 13, ["马岭岗镇", "吕陵镇", "安兴镇", "大黄集镇"], base_lon - 0.25, base_lat - 0.18, (1, 2), (2, 3), (35, 95), (0.16, 0.30), (0.04, 0.08)),
        ("北部片区", "乡镇", 13, ["吴店镇", "都司镇", "王浩屯镇", "李村镇"], base_lon + 0.03, base_lat + 0.31, (2, 4), (2, 4), (40, 105), (0.08, 0.18), (0.12, 0.24)),
    ]

    rows = []
    outlet_no = 1
    for risk_zone, region_type, count, towns, lon_center, lat_center, road_range, service_range, order_range, timeout_range, return_range in segments:
        for idx in range(count):
            town = str(rng.choice(towns))
            rows.append(
                {
                    "outlet_id": f"O{outlet_no:03d}",
                    "outlet_name": f"{town}{idx + 1}号配送网点",
                    "town": town,
                    "region_type": region_type,
                    "lon": round(lon_center + float(rng.normal(0, 0.04 if region_type == "乡镇" else 0.02)), 6),
                    "lat": round(lat_center + float(rng.normal(0, 0.04 if region_type == "乡镇" else 0.02)), 6),
                    "road_level": int(rng.integers(road_range[0], road_range[1] + 1)),
                    "service_level": int(rng.integers(service_range[0], service_range[1] + 1)),
                    "daily_orders": int(rng.integers(order_range[0], order_range[1] + 1)),
                    "timeout_rate": round(float(rng.uniform(*timeout_range)), 4),
                    "return_rate": round(float(rng.uniform(*return_range)), 4),
                    "risk_zone": risk_zone,
                }
            )
            outlet_no += 1

    outlets_with_meta = pd.DataFrame(rows)
    outlets = outlets_with_meta.drop(columns=["risk_zone"])
    return outlets, outlets_with_meta.set_index("outlet_id")


def generate_inventory(products: pd.DataFrame, product_profile: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    product_info = products.set_index("product_id")
    weights = product_profile["sales_weight"]
    high_sales = set(weights.nlargest(45).index)
    low_sales = set(weights.nsmallest(65).index)

    rows = []
    for idx, product_id in enumerate(products["product_id"], start=1):
        cost_price = float(product_info.at[product_id, "cost_price"])
        category = str(product_info.at[product_id, "category"])
        if product_id in high_sales:
            safety_stock = int(rng.integers(80, 260))
            stock_qty = int(rng.integers(15, 80)) if rng.random() < 0.28 else int(rng.integers(160, 520))
            last_in_days = int(rng.integers(1, 45))
        elif product_id in low_sales or category in LOW_FREQ_CATEGORIES:
            safety_stock = int(rng.integers(12, 80))
            stock_qty = int(rng.integers(260, 1200)) if rng.random() < 0.45 else int(rng.integers(40, 240))
            last_in_days = int(rng.integers(45, 180))
        else:
            safety_stock = int(rng.integers(40, 150))
            stock_qty = int(rng.integers(80, 420))
            last_in_days = int(rng.integers(7, 100))

        warehouse_id = "W001" if idx % 3 != 0 else "W002"
        rows.append(
            {
                "snapshot_date": END_DATE.date().isoformat(),
                "product_id": product_id,
                "warehouse_id": warehouse_id,
                "location_code": f"{warehouse_id}-A{int(rng.integers(1, 20)):02d}-{int(rng.integers(1, 80)):02d}",
                "stock_qty": stock_qty,
                "stock_amount": round(stock_qty * cost_price, 2),
                "safety_stock": safety_stock,
                "last_in_date": (END_DATE - pd.Timedelta(days=last_in_days)).date().isoformat(),
            }
        )

    return pd.DataFrame(rows)


def sample_quantity(category: str, rng: np.random.Generator) -> int:
    if category in FRESH_CATEGORIES:
        return int(rng.choice([1, 2, 3, 4, 5, 6, 8, 10], p=[0.18, 0.24, 0.18, 0.13, 0.10, 0.08, 0.06, 0.03]))
    if category == "零食饮料":
        return int(rng.choice([1, 2, 3, 4, 6, 8, 12], p=[0.16, 0.22, 0.18, 0.16, 0.13, 0.10, 0.05]))
    if category == "日用品":
        return int(rng.choice([1, 2, 3, 4, 6], p=[0.28, 0.28, 0.20, 0.14, 0.10]))
    return int(rng.choice([1, 2, 3], p=[0.70, 0.23, 0.07]))


def generate_orders(products: pd.DataFrame, product_profile: pd.DataFrame, outlets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    product_info = products.set_index("product_id")
    product_ids = product_profile.index.to_numpy()
    product_probs = product_profile["sales_weight"].to_numpy(dtype=float)
    product_probs = product_probs / product_probs.sum()

    outlet_ids = outlets["outlet_id"].to_numpy()
    outlet_probs = outlets["daily_orders"].to_numpy(dtype=float)
    outlet_probs = outlet_probs / outlet_probs.sum()

    order_dates = pd.date_range(END_DATE - pd.Timedelta(days=89), END_DATE, freq="D")
    rows = []
    for idx in range(UNIQUE_ORDER_ROWS):
        product_id = str(rng.choice(product_ids, p=product_probs))
        product = product_info.loc[product_id]
        quantity = sample_quantity(str(product["category"]), rng)
        urgent_prob = 0.08 + 0.06 * int(product["is_fresh"]) + 0.04 * int(product["is_key_product"])
        rows.append(
            {
                "order_id": f"ORD{idx + 1:06d}",
                "order_date": pd.Timestamp(rng.choice(order_dates)).date().isoformat(),
                "customer_id": f"C{int(rng.integers(1, 1801)):05d}",
                "product_id": product_id,
                "quantity": quantity,
                "order_amount": round(quantity * float(product["unit_price"]), 2),
                "outlet_id": str(rng.choice(outlet_ids, p=outlet_probs)),
                "order_status": "已签收",
                "is_urgent": int(rng.random() < urgent_prob),
            }
        )

    orders = pd.DataFrame(rows)
    duplicate_source_idx = rng.choice(orders.index.to_numpy(), size=12, replace=False)
    duplicate_rows = orders.loc[duplicate_source_idx].copy()
    duplicate_order_ids = set(duplicate_rows["order_id"].tolist())
    orders = pd.concat([orders, duplicate_rows], ignore_index=True)
    orders = orders.sample(frac=1, random_state=SEED).reset_index(drop=True)

    missing_candidates = orders.loc[~orders["order_id"].isin(duplicate_order_ids)].drop_duplicates("order_id").index.to_numpy()
    missing_idx = rng.choice(missing_candidates, size=6, replace=False)
    orders.loc[missing_idx, "quantity"] = np.nan

    format_candidates = np.setdiff1d(missing_candidates, missing_idx)
    format_idx = rng.choice(format_candidates, size=8, replace=False)
    format_dates = pd.to_datetime(orders.loc[format_idx, "order_date"])
    orders.loc[format_idx, "order_date"] = [f"{item.year}/{item.month}/{item.day}" for item in format_dates]
    return orders


def generate_delivery(orders: pd.DataFrame, outlets: pd.DataFrame, outlet_meta: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    outlets_by_id = outlets.set_index("outlet_id")
    rows = []
    for idx, order in orders.iterrows():
        outlet_id = str(order["outlet_id"])
        outlet = outlets_by_id.loc[outlet_id]
        region_type = str(outlet["region_type"])
        planned_hours = 24 if region_type == "城区" else 48
        if int(order["is_urgent"]) == 1:
            planned_hours = max(12, planned_hours - 12)

        base_hours = float(rng.normal(16 if region_type == "城区" else 38, 5 if region_type == "城区" else 10))
        road_delay = (5 - int(outlet["road_level"])) * 1.8
        service_delay = (5 - int(outlet["service_level"])) * 1.2
        actual_hours = round(max(4, base_hours + road_delay + service_delay), 1)
        order_date = pd.to_datetime(order["order_date"], errors="coerce")
        if pd.isna(order_date):
            order_date = END_DATE
        ship_time = order_date + pd.Timedelta(hours=int(rng.integers(7, 20)), minutes=int(rng.integers(0, 60)))
        sign_time = ship_time + pd.Timedelta(hours=actual_hours)
        rows.append(
            {
                "delivery_id": f"D{idx + 1:06d}",
                "order_id": order["order_id"],
                "outlet_id": outlet_id,
                "ship_time": ship_time,
                "sign_time": sign_time,
                "planned_hours": planned_hours,
                "actual_hours": actual_hours,
                "is_timeout": 0,
                "delivery_cost": 0.0,
                "weather": str(rng.choice(["晴", "雨", "雾", "大风"], p=[0.68, 0.16, 0.08, 0.08])),
            }
        )

    delivery = pd.DataFrame(rows)
    delivery["actual_hours"] = delivery["actual_hours"].clip(lower=4, upper=68)
    delivery["sign_time"] = delivery["ship_time"] + pd.to_timedelta(delivery["actual_hours"], unit="h")
    delivery["is_timeout"] = (delivery["actual_hours"] > delivery["planned_hours"]).astype(int)

    weather_add = {"晴": 0.0, "雨": 3.5, "雾": 4.5, "大风": 4.0}
    costs = []
    for _, row in delivery.iterrows():
        outlet = outlets_by_id.loc[row["outlet_id"]]
        base_cost = 5 + max(float(row["actual_hours"]), 1) * 0.55
        road_cost = (5 - int(outlet["road_level"])) * 1.4
        town_cost = 3.0 if outlet["region_type"] == "乡镇" else 0.0
        costs.append(round(base_cost + road_cost + town_cost + weather_add[str(row["weather"])], 2))
    delivery["delivery_cost"] = costs

    delivery["ship_time"] = pd.to_datetime(delivery["ship_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    delivery["sign_time"] = delivery["sign_time"].apply(lambda value: "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S"))
    return delivery


def generate_returns(orders: pd.DataFrame, products: pd.DataFrame, outlets: pd.DataFrame, delivery: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    product_info = products.set_index("product_id")
    outlet_info = outlets.set_index("outlet_id")
    delivery_info = delivery.set_index("delivery_id")
    scores = []
    for idx, order in orders.iterrows():
        product = product_info.loc[order["product_id"]]
        outlet = outlet_info.loc[order["outlet_id"]]
        delivery_row = delivery_info.iloc[idx]
        score = 1.0 + float(outlet["return_rate"]) * 10
        if int(delivery_row["is_timeout"]) == 1:
            score += 2.5
        if int(product["is_fresh"]) == 1 and int(delivery_row["is_timeout"]) == 1:
            score += 2.5
        if str(product["category"]) in LOW_FREQ_CATEGORIES:
            score += 1.2
        scores.append(score)

    probabilities = np.array(scores, dtype=float)
    probabilities = probabilities / probabilities.sum()
    selected = rng.choice(orders.index.to_numpy(), size=RETURN_ROWS, replace=False, p=probabilities)
    reasons = ["配送超时", "商品破损", "客户拒收", "质量问题", "地址错误"]
    rows = []
    for idx, order_idx in enumerate(selected):
        order = orders.loc[order_idx]
        product = product_info.loc[order["product_id"]]
        reason_probs = [0.32, 0.20, 0.18, 0.18, 0.12]
        if str(product["category"]) in LOW_FREQ_CATEGORIES:
            reason_probs = [0.18, 0.18, 0.18, 0.34, 0.12]
        return_date = min(pd.Timestamp(order["order_date"]) + pd.Timedelta(days=int(rng.integers(1, 10))), END_DATE)
        rows.append(
            {
                "return_id": f"R{idx + 1:05d}",
                "order_id": order["order_id"],
                "product_id": order["product_id"],
                "return_date": return_date.date().isoformat(),
                "return_reason": str(rng.choice(reasons, p=reason_probs)),
                "refund_amount": round(float(order["order_amount"]) * float(rng.uniform(0.82, 1.0)), 2),
            }
        )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, file_name: str) -> None:
    df.to_csv(RAW_DIR / file_name, index=False, encoding="utf-8-sig")


def generate_all() -> dict[str, int]:
    ensure_dirs()
    cleanup_previous_raw_files()
    rng = np.random.default_rng(SEED)
    products, product_profile = generate_products(rng)
    outlets, outlet_meta = generate_outlets(rng)
    inventory = generate_inventory(products, product_profile, rng)
    orders = generate_orders(products, product_profile, outlets, rng)
    delivery = generate_delivery(orders, outlets, outlet_meta, rng)
    returns = generate_returns(orders, products, outlets, delivery, rng)

    tables = {
        "orders.csv": orders,
        "delivery.csv": delivery,
        "inventory.csv": inventory,
        "products.csv": products,
        "outlets.csv": outlets,
        "returns.csv": returns,
    }
    for file_name, dataframe in tables.items():
        write_csv(dataframe, file_name)
    return {file_name: len(dataframe) for file_name, dataframe in tables.items()}


def main() -> None:
    counts = generate_all()
    print(f"Mock data generated in: {RAW_DIR}")
    for file_name, row_count in counts.items():
        print(f"{file_name}: {row_count}")


if __name__ == "__main__":
    main()
