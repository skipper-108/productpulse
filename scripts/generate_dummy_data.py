from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

BASE = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE.mkdir(parents=True, exist_ok=True)

n_users = 1_000
user_ids = np.arange(1, n_users + 1)

orders, events = [], []
order_id = 1000

for u in user_ids:
    n_o = np.random.randint(1, 6)  # 1–5 orders per user
    for _ in range(n_o):
        date = fake.date_between(start_date="-180d", end_date="today")
        revenue = round(np.random.exponential(80) + 10, 2)
        orders.append([u, order_id, date, revenue])
        order_id += 1

        # three events within 0–3 days of order date
        for et in ("view", "add_to_cart", "checkout"):
            evt_time = fake.date_time_between(
                start_date=date,
                end_date=date + timedelta(days=3),
            )
            events.append([len(events) + 1, u, et, evt_time.isoformat()])

pd.DataFrame(
    orders, columns=["user_id", "order_id", "order_date", "revenue"]
).to_csv(BASE / "orders.csv", index=False)

pd.DataFrame(
    events, columns=["event_id", "user_id", "event_type", "event_time"]
).to_csv(BASE / "events.csv", index=False)

print("✔ Dummy data generated:", (BASE / 'orders.csv').stat().st_size, "bytes")
