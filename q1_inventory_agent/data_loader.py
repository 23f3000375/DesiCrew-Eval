"""Load the messy inventory workbook (title rows above the header, newlines in header names)."""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "Inventory-Records-Sample-Data.xlsx"

COLUMN_DOCS = {
    "product_id": "Product ID, e.g. P101 (unique)",
    "product_name": "Product name",
    "opening_stock": "Units in stock at the start of the period",
    "units_purchased": "Units purchased / stocked in during the period ('Purchase/Stock in')",
    "units_sold": "Units sold during the period",
    "hand_in_stock": "Units on hand now, as recorded in the sheet ('Hand-In-Stock')",
    "cost_per_unit_usd": "Cost price per unit in USD",
    "total_cost_usd": "Total cost value in USD as recorded in the sheet ('Cost Price Total')",
}


def _snake(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).replace("\n", " ")).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def load_inventory(path=DATA) -> pd.DataFrame:
    """Find the header row by looking for 'Product ID', then read the table below it."""
    raw = pd.read_excel(path, header=None)
    hdr = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.contains("Product ID", case=False).any())
    first_col = next(j for j in range(raw.shape[1]) if "product id" in str(raw.iloc[hdr, j]).lower())
    df = raw.iloc[hdr + 1:, first_col:].copy()
    df.columns = list(COLUMN_DOCS)  # same order as the sheet; asserted below
    heads = [_snake(x) for x in raw.iloc[hdr, first_col:first_col + len(COLUMN_DOCS)]]
    assert heads[:2] == ["product_id", "product_name"] and "hand_in" in heads[5], f"unexpected header: {heads}"
    df = df.dropna(how="all")
    for c in list(COLUMN_DOCS)[2:]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df.reset_index(drop=True)


def schema_text(df: pd.DataFrame) -> str:
    lines = [f"DataFrame `df`: {len(df)} rows x {df.shape[1]} columns (one row per product)."]
    for c, d in COLUMN_DOCS.items():
        lines.append(f"- {c} ({df[c].dtype}): {d}")
    lines.append("\nFirst rows:\n" + df.head(5).to_string(index=False))
    return "\n".join(lines)


if __name__ == "__main__":
    d = load_inventory()
    print(schema_text(d))
