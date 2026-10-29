"""
AGRICONNECT - Price Predictor management commands.

  python manage_prices.py seed-demo            # (re)load synthetic demo prices
  python manage_prices.py load-csv FILE.csv    # replace data with real Agmarknet prices
  python manage_prices.py train                # retrain the model and print the metrics

Typical flow for real data:
  1. Download daily mandi prices (State, Commodity, Arrival_Date, Modal_Price) as CSV
  2. python manage_prices.py load-csv data/prices.csv
  3. python manage_prices.py train
Retrain whenever you add newer data (e.g. monthly).
"""

import sys

from app import app
from models import db, PriceHistory
from price_data import generate_demo_rows, load_csv_rows
from price_engine import train_all


def _replace_rows(rows):
    PriceHistory.query.delete()
    db.session.bulk_save_objects([PriceHistory(**r) for r in rows])
    db.session.commit()
    print(f"Loaded {len(rows)} weekly price rows.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    with app.app_context():
        if cmd == 'seed-demo':
            _replace_rows(generate_demo_rows())
            train_all()
        elif cmd == 'load-csv' and len(sys.argv) >= 3:
            _replace_rows(load_csv_rows(sys.argv[2]))
            train_all()
        elif cmd == 'train':
            train_all()
        else:
            print(__doc__)


if __name__ == '__main__':
    main()
