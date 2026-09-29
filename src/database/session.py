"""
Database connection, session management, and data seeder.
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from datetime import datetime

from src.config import settings
from src.database.models import Base, Store, DailySale, InventoryItem

# Handle SQLite multi-threading connection check
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized.")

def get_db():
    """Dependency generator for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_database_from_csv(
    raw_data_dir: Path = settings.RAW_DATA_DIR,
    processed_file: Path = settings.PROCESSED_DATA_DIR / "clean_sales.csv"
):
    """
    Seeds stores, initial inventory, and recent sales into database tables.
    """
    init_db()
    db = SessionLocal()
    try:
        # 1. Seed Stores
        stores_file = raw_data_dir / "stores.csv"
        if stores_file.exists() and db.query(Store).count() == 0:
            stores_df = pd.read_csv(stores_file)
            for _, row in stores_df.iterrows():
                store = Store(
                    store_nbr=int(row["store_nbr"]),
                    city=str(row["city"]),
                    state=str(row["state"]),
                    type=str(row["type"]),
                    cluster=int(row["cluster"])
                )
                db.add(store)
            db.commit()
            print(f"Seeded {len(stores_df)} stores into database.")
            
        # 2. Seed Inventory
        inv_file = raw_data_dir / "inventory.csv"
        if inv_file.exists() and db.query(InventoryItem).count() == 0:
            inv_df = pd.read_csv(inv_file)
            for _, row in inv_df.iterrows():
                inv = InventoryItem(
                    product_id=str(row["product_id"]),
                    store_id=int(row["store_id"]),
                    current_stock=float(row["current_stock"]),
                    reserved_stock=float(row.get("reserved_stock", 0.0)),
                    on_order_stock=float(row.get("on_order_stock", 0.0)),
                    lead_time_days=int(row.get("lead_time_days", 7)),
                    safety_stock=float(row.get("safety_stock", 20.0)),
                    unit_cost=float(row.get("unit_cost", 10.0)),
                    selling_price=float(row.get("selling_price", 15.0)),
                    reorder_multiple=int(row.get("reorder_multiple", 5)),
                    min_order_qty=int(row.get("min_order_qty", 10)),
                    last_updated=datetime.strptime(str(row["last_updated"]), "%Y-%m-%d").date()
                )
                db.add(inv)
            db.commit()
            print(f"Seeded {len(inv_df)} inventory records into database.")
            
        # 3. Seed Recent Sales (recent 90 days for snappy SQL analytics)
        if processed_file.exists() and db.query(DailySale).count() == 0:
            sales_df = pd.read_csv(processed_file)
            sales_df["date"] = pd.to_datetime(sales_df["date"])
            recent_cutoff = sales_df["date"].max() - pd.Timedelta(days=90)
            recent_sales = sales_df[sales_df["date"] >= recent_cutoff]
            
            sales_objects = []
            for _, row in recent_sales.iterrows():
                sales_objects.append(DailySale(
                    date=row["date"].date(),
                    store_nbr=int(row["store_nbr"]),
                    family=str(row["family"]),
                    sales=float(row["sales"]),
                    onpromotion=int(row.get("onpromotion", 0)),
                    oil_price=float(row.get("oil_price", 70.0)),
                    is_holiday=int(row.get("is_holiday", 0))
                ))
            db.bulk_save_objects(sales_objects)
            db.commit()
            print(f"Seeded {len(sales_objects):,} recent daily sales into database.")
            
    finally:
        db.close()
