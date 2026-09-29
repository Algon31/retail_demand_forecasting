"""
Dataset Generator: Generates realistic store sales, calendar, holiday, oil,
and inventory datasets with retail time-series dynamics (weekly seasonality,
promotional lifts, holiday surges, store heterogeneity, and inventory states).
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.config import settings

PRODUCT_FAMILIES = [
    "BEVERAGES",
    "BREAD/BAKERY",
    "CLEANING",
    "DAIRY",
    "GROCERY I",
    "MEATS",
    "PERSONAL CARE",
    "POULTRY",
    "PRODUCE",
    "DELI"
]

STORE_CITIES = [
    (1, "Quito", "Pichincha", "D", 13),
    (2, "Quito", "Pichincha", "D", 13),
    (3, "Quito", "Pichincha", "D", 8),
    (4, "Guayaquil", "Guayas", "B", 9),
    (5, "Santo Domingo", "Santo Domingo de los Tsachilas", "D", 4),
    (6, "Cuenca", "Azuay", "D", 2),
    (7, "Ambato", "Tungurahua", "A", 14),
    (8, "Machala", "El Oro", "D", 8),
    (9, "Quito", "Pichincha", "B", 6),
    (10, "Guayaquil", "Guayas", "B", 10),
]

def generate_synthetic_data(
    start_date: str = "2024-01-01",
    end_date: str = "2026-06-30",
    output_dir: Path = settings.RAW_DATA_DIR,
    seed: int = 42
):
    """
    Generates synthetic datasets following Kaggle Store Sales schema plus inventory table.
    """
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    
    # 1. Stores Metadata
    stores_df = pd.DataFrame(
        STORE_CITIES,
        columns=["store_nbr", "city", "state", "type", "cluster"]
    )
    stores_df.to_csv(output_dir / "stores.csv", index=False)
    
    # 2. Oil Prices (Macro factor with random walk)
    oil_prices = [75.0]
    for _ in range(n_days - 1):
        change = np.random.normal(0, 0.8)
        oil_prices.append(max(40.0, min(120.0, oil_prices[-1] + change)))
    oil_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "dcoilwtico": oil_prices})
    # Add weekend / holiday missing values as in real Kaggle dataset
    oil_df.loc[dates.dayofweek >= 5, "dcoilwtico"] = np.nan
    oil_df.to_csv(output_dir / "oil.csv", index=False)
    
    # 3. Holidays & Events
    holidays = []
    for dt in dates:
        # New Year
        if dt.month == 1 and dt.day == 1:
            holidays.append({"date": dt.strftime("%Y-%m-%d"), "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Primer dia del ano", "transferred": False})
        # Carnival (approx Feb 15)
        elif dt.month == 2 and dt.day == 15:
            holidays.append({"date": dt.strftime("%Y-%m-%d"), "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Carnaval", "transferred": False})
        # Labor Day (May 1)
        elif dt.month == 5 and dt.day == 1:
            holidays.append({"date": dt.strftime("%Y-%m-%d"), "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Dia del Trabajo", "transferred": False})
        # Christmas (Dec 25)
        elif dt.month == 12 and dt.day == 25:
            holidays.append({"date": dt.strftime("%Y-%m-%d"), "type": "Holiday", "locale": "National", "locale_name": "Ecuador", "description": "Navidad", "transferred": False})
        # Black Friday / Cyber Monday promo events
        elif dt.month == 11 and dt.day == 26:
            holidays.append({"date": dt.strftime("%Y-%m-%d"), "type": "Event", "locale": "National", "locale_name": "Ecuador", "description": "Black Friday", "transferred": False})
            
    holidays_df = pd.DataFrame(holidays)
    if holidays_df.empty:
        holidays_df = pd.DataFrame(columns=["date", "type", "locale", "locale_name", "description", "transferred"])
    holidays_df.to_csv(output_dir / "holidays_events.csv", index=False)
    
    holiday_dates = set(holidays_df["date"].tolist())
    
    # Base category volumes and multipliers
    base_category_volume = {
        "BEVERAGES": 450.0,
        "GROCERY I": 600.0,
        "CLEANING": 220.0,
        "DAIRY": 180.0,
        "PRODUCE": 280.0,
        "BREAD/BAKERY": 130.0,
        "POULTRY": 95.0,
        "MEATS": 85.0,
        "DELI": 65.0,
        "PERSONAL CARE": 75.0,
    }
    
    # Store size scale factors
    store_scale = {s[0]: 0.7 + 0.08 * s[4] for s in STORE_CITIES}
    
    records = []
    transactions_list = []
    record_id = 1
    
    for dt in dates:
        dt_str = dt.strftime("%Y-%m-%d")
        day_of_week = dt.dayofweek
        # Weekend boost (Saturdays & Sundays)
        weekend_mult = 1.35 if day_of_week in [5, 6] else 1.0
        # Holiday boost
        holiday_mult = 1.5 if dt_str in holiday_dates else 1.0
        
        for store_nbr in stores_df["store_nbr"]:
            store_mult = store_scale[store_nbr]
            store_tx_count = int(np.random.normal(1200 * store_mult * weekend_mult, 80))
            transactions_list.append({"date": dt_str, "store_nbr": store_nbr, "transactions": max(100, store_tx_count)})
            
            for family in PRODUCT_FAMILIES:
                base_vol = base_category_volume[family] * store_mult
                
                # Promotion probability (~15% of days have promotions)
                is_promo = np.random.rand() < 0.18
                promo_count = np.random.randint(1, 15) if is_promo else 0
                promo_mult = 1.0 + (0.04 * promo_count) if is_promo else 1.0
                
                # Trend component (slight upward demand over time)
                days_from_start = (dt - dates[0]).days
                trend_mult = 1.0 + (0.0002 * days_from_start)
                
                # Seasonality: day of week modulation
                dow_weights = [0.85, 0.88, 0.92, 0.95, 1.15, 1.40, 1.25]
                dow_mult = dow_weights[day_of_week]
                
                # Target sales with lognormal noise
                expected_sales = base_vol * trend_mult * dow_mult * promo_mult * holiday_mult
                sales_noise = np.random.lognormal(mean=0.0, sigma=0.15)
                actual_sales = round(max(0.0, expected_sales * sales_noise), 2)
                
                records.append({
                    "id": record_id,
                    "date": dt_str,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": actual_sales,
                    "onpromotion": promo_count
                })
                record_id += 1
                
    train_df = pd.DataFrame(records)
    train_df.to_csv(output_dir / "train.csv", index=False)
    
    tx_df = pd.DataFrame(transactions_list)
    tx_df.to_csv(output_dir / "transactions.csv", index=False)
    
    # 4. Inventory Data (Operational store-level inventory snapshot)
    # Unit costs and retail margins
    category_economics = {
        "BEVERAGES": (3.50, 5.00, 3, 25),
        "GROCERY I": (2.20, 3.80, 5, 30),
        "CLEANING": (4.10, 6.50, 4, 15),
        "DAIRY": (1.80, 2.70, 2, 20),
        "PRODUCE": (1.20, 2.00, 1, 15),
        "BREAD/BAKERY": (1.50, 2.50, 2, 10),
        "POULTRY": (5.00, 7.50, 3, 10),
        "MEATS": (7.00, 10.50, 4, 10),
        "DELI": (6.00, 9.00, 2, 10),
        "PERSONAL CARE": (3.00, 5.50, 7, 20),
    }
    
    inventory_records = []
    latest_date = dates[-1].strftime("%Y-%m-%d")
    
    for store_nbr in stores_df["store_nbr"]:
        for family in PRODUCT_FAMILIES:
            unit_cost, selling_price, lead_time, safety_base = category_economics[family]
            daily_avg = (train_df[(train_df["store_nbr"] == store_nbr) & (train_df["family"] == family)]["sales"].tail(30).mean())
            
            # Stock levels: some are critically low (dead/out-of-stock risk), some healthy, some overstocked
            stock_ratio = np.random.choice([0.2, 0.5, 1.1, 1.8, 2.5], p=[0.15, 0.25, 0.35, 0.15, 0.10])
            current_stock = round(daily_avg * (lead_time + 3) * stock_ratio, 1)
            reserved_stock = round(daily_avg * 0.5 * np.random.uniform(0.1, 0.8), 1)
            on_order_stock = round(daily_avg * lead_time * (1 if stock_ratio < 0.6 else 0), 1)
            safety_stock = round(max(5.0, safety_base * store_scale[store_nbr]), 1)
            
            inventory_records.append({
                "product_id": family,
                "store_id": int(store_nbr),
                "current_stock": max(0.0, current_stock),
                "reserved_stock": reserved_stock,
                "on_order_stock": on_order_stock,
                "lead_time_days": lead_time,
                "safety_stock": safety_stock,
                "unit_cost": unit_cost,
                "selling_price": selling_price,
                "reorder_multiple": 5,
                "min_order_qty": 10,
                "last_updated": latest_date
            })
            
    inventory_df = pd.DataFrame(inventory_records)
    inventory_df.to_csv(output_dir / "inventory.csv", index=False)
    
    print(f"Generated synthetic retail data in {output_dir}:")
    print(f" - train.csv: {len(train_df):,} records ({start_date} to {end_date})")
    print(f" - stores.csv: {len(stores_df)} stores")
    print(f" - oil.csv: {len(oil_df)} records")
    print(f" - holidays_events.csv: {len(holidays_df)} events")
    print(f" - transactions.csv: {len(tx_df)} records")
    print(f" - inventory.csv: {len(inventory_df)} SKUs/store records")
    
    return {
        "train": train_df,
        "stores": stores_df,
        "oil": oil_df,
        "holidays": holidays_df,
        "transactions": tx_df,
        "inventory": inventory_df
    }

if __name__ == "__main__":
    generate_synthetic_data()
