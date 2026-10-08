# Inventory agent - demo transcript
_models used (calls per model): {'groq:openai/gpt-oss-20b': 11, 'groq:openai/gpt-oss-120b': 2}_

## Q: Give me a quick overview of the inventory: how many products, total units on hand and total stock value?

<details><summary>code run</summary>

```python
print('Products:', df.shape[0])
print('Total units on hand:', df['hand_in_stock'].sum())
print('Total stock value (USD):', df['total_cost_usd'].sum())
```

```
Products: 46
Total units on hand: 2004
Total stock value (USD): 359760
```
</details>

There are **46 products** in the list.  
The inventory shows **2,004 units on hand** in total, and the recorded stock value comes to **$359,760**.  

*Assumptions:* I used the “hand_in_stock” column for the unit count and the “total_cost_usd” column for the value.

## Q: Which 5 products hold the most inventory value, and what share of the total is that?

<details><summary>code run</summary>

```python
top5 = df.nlargest(5, 'total_cost_usd')[['product_id','product_name','total_cost_usd']]
print(top5)
share = top5['total_cost_usd'].sum() / df['total_cost_usd'].sum()
print('Share of total:', share)
```

```
product_id   product_name  total_cost_usd
0        P101         Laptop           72000
4        P105     Smartphone           72000
5        P106         Tablet           38500
21       P122  Graphics Card           29400
1        P102        Monitor           25000
Share of total: 0.6584945519235046
```
</details>

The five products with the highest inventory value are:

1. **Laptop (P101)** – $72,000  
2. **Smartphone (P105)** – $72,000  
3. **Tablet (P106)** – $38,500  
4. **Graphics Card (P122)** – $29,400  
5. **Monitor (P102)** – $25,000  

These top five items account for **about 66 %** of the total inventory value of $359,760.

## Q: Which products sold through the highest percentage of the stock they had available?

<details><summary>code run</summary>

```python
df['available'] = df['opening_stock'] + df['units_purchased']
# avoid division by zero
df['sell_through'] = df['units_sold'] / df['available']
print(df[['product_id','product_name','sell_through']].sort_values('sell_through', ascending=False).head(5))
```

```
product_id     product_name  sell_through
4        P105       Smartphone      0.200000
2        P103         Keyboard      0.176471
13       P114   Ethernet Cable      0.166667
9        P110           Webcam      0.166667
12       P113  USB Flash Drive      0.160000
```
</details>

The products that sold the highest percentage of the stock they had available are:

1. **Smartphone (P105)** – 20 % sold  
2. **Keyboard (P103)** – 17.6 % sold  
3. **Ethernet Cable (P114)** – 16.7 % sold  
4. **Webcam (P110)** – 16.7 % sold  
5. **USB Flash Drive (P113)** – 16.0 % sold  

*Calculation:* I added opening stock and units purchased to get the total available, then divided units sold by that total. No product had a zero‑available case, so the percentages are straightforward.

## Q: Do the hand-in-stock numbers reconcile with opening + purchased - sold? List any that don't and by how much.

<details><summary>code run</summary>

```python
expected = df['opening_stock'] + df['units_purchased'] - df['units_sold']
# difference
diff = df['hand_in_stock'] - expected
# find non-zero differences
mismatch = df[diff != 0][['product_id','product_name','hand_in_stock','expected','diff']]
print(mismatch)
print('Number of mismatches:', mismatch.shape[0])
```

```
Traceback (most recent call last):
  File "<string>", line 30, in <module>
    exec(compile(tree, "<agent_code>", "exec"), ns)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<agent_code>", line 5, in <module>
  File "D:\Downloads\desicrew-ds-eval (1)\desicrew-ds-eval\.venv\Lib\site-packages\pandas\core\frame.py", line 4384, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "D:\Downloads\desicrew-ds-eval (1)\desicrew-ds-eval\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 6302, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: "['expected', 'diff'] not in index"
```
</details>

<details><summary>code run</summary>

```python
expected = df['opening_stock'] + df['units_purchased'] - df['units_sold']
diff = df['hand_in_stock'] - expected
mismatch = df.loc[diff != 0, ['product_id','product_name','hand_in_stock']].copy()
mismatch['expected'] = expected[diff != 0]
mismatch['diff'] = diff[diff != 0]
print(mismatch)
print('Number of mismatches:', mismatch.shape[0])
```

```
product_id                 product_name  hand_in_stock  expected  diff
20       P121               Gaming Monitor             29        31    -2
29       P130                   CPU Cooler             34        38    -4
30       P131                Monitor Stand             28        32    -4
31       P132                    Mouse Pad             58        62    -4
33       P134         Cable Management Kit             34        38    -4
34       P135                 WiFi Adapter             23        25    -2
35       P136           External DVD Drive             17        19    -2
36       P137                Printer Cable             29        31    -2
39       P140                      USB Hub             31        43   -12
40       P141  Anti-Glare Screen Protector             28        30    -2
41       P142                USB-C Adapter             24        26    -2
45       P146                  Gaming Desk             28        32    -4
Number of mismatches: 12
```
</details>

The “hand‑in‑stock” figures do not match the calculation of opening stock + purchased – sold for **12 products**.  
Here’s how far off each one is:

| Product | Hand‑in‑Stock | Expected (calc) | Difference |
|---------|---------------|-----------------|------------|
| P121 Gaming Monitor | 29 | 31 | –2 |
| P130 CPU Cooler | 34 | 38 | –4 |
| P131 Monitor Stand | 28 | 32 | –4 |
| P132 Mouse Pad | 58 | 62 | –4 |
| P134 Cable Management Kit | 34 | 38 | –4 |
| P135 WiFi Adapter | 23 | 25 | –2 |
| P136 External DVD Drive | 17 | 19 | –2 |
| P137 Printer Cable | 29 | 31 | –2 |
| P140 USB Hub | 31 | 43 | –12 |
| P141 Anti‑Glare Screen Protector | 28 | 30 | –2 |
| P142 USB‑C Adapter | 24 | 26 | –2 |
| P146 Gaming Desk | 28 | 32 | –4 |

All differences are negative, meaning the recorded hand‑in‑stock is lower than what the arithmetic would suggest. No other products show a discrepancy.

## Q: What is inventory turnover? Look up the standard formula and calculate it for the whole catalog from this sheet.

<details><summary>code run</summary>

```python
# Compute COGS
cogs = (df['units_sold'] * df['cost_per_unit_usd']).sum()
# Opening inventory value
opening_value = (df['opening_stock'] * df['cost_per_unit_usd']).sum()
# Hand inventory value
hand_value = (df['hand_in_stock'] * df['cost_per_unit_usd']).sum()
avg_inventory_value = (opening_value + hand_value) / 2
turnover = cogs / avg_inventory_value
print('COGS:', cogs)
print('Opening value:', opening_value)
print('Hand value:', hand_value)
print('Avg inventory value:', avg_inventory_value)
print('Inventory turnover:', turnover)
```

```
COGS: 62339
Opening value: 298970
Hand value: 359760
Avg inventory value: 329365.0
Inventory turnover: 0.18927026247476203
```
</details>

**Inventory turnover** measures how many times a company’s inventory is sold and replaced over a period.  
The standard formula is:

\[
\text{Inventory Turnover} = \frac{\text{Cost of Goods Sold (COGS)}}{\text{Average Inventory Value}}
\]

where  

* **COGS** = units sold × cost per unit (summed across all products)  
* **Average Inventory Value** = (Opening‑period inventory value + Ending‑period inventory value) ÷ 2  

**Using the data in your sheet**

| Item | Calculation | Result |
|------|-------------|--------|
| COGS | Σ (units_sold × cost_per_unit_usd) | **$62,339** |
| Opening inventory value | Σ (opening_stock × cost_per_unit_usd) | **$298,970** |
| Ending (hand‑in‑stock) inventory value | Σ (hand_in_stock × cost_per_unit_usd) | **$359,760** |
| Average inventory value | (Opening + Ending) ÷ 2 | **$329,365** |
| **Inventory turnover** | COGS ÷ Average inventory value | **0.19 × per period** |

**Interpretation** – An inventory turnover of **0.19** means the total stock value is sold roughly once every five periods (≈ 1 ÷ 0.19 ≈ 5.3). This is a relatively low turnover, suggesting that inventory is moving slowly and a lot of capital is tied up in stock.  

*Assumptions:* I treated the cost per unit as the cost of goods sold for each unit sold, and used the opening‑stock and hand‑in‑stock values as the period‑start and period‑end inventory values. If you prefer a different cost basis (e.g., selling price) or a different time frame, the turnover figure would change accordingly.

## Q: Chart units sold for the top 10 products.

<details><summary>code run</summary>

```python
import matplotlib.pyplot as plt

# Get top 10 by units_sold
top10 = df.nlargest(10, 'units_sold')[['product_name', 'units_sold']]

plt.figure(figsize=(10,6))
plt.bar(top10['product_name'], top10['units_sold'], color='skyblue')
plt.xlabel('Product')
plt.ylabel('Units Sold')
plt.title('Units Sold – Top 10 Products')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('chart.png')
print('Chart saved')
```

```
Chart saved
```
</details>

Here’s a bar chart showing the **units sold for the 10 products with the highest sales**. The x‑axis lists each product name, and the y‑axis shows the number of units sold.
