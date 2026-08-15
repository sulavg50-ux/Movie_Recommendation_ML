import csv
import pandas as pd

rows = []
with open('data/movies.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    while True:
        try:
            row = next(reader)
            rows.append(row)
        except StopIteration:
            # Normal end of file
            break
        except csv.Error as e:
            # Usually "unexpected end of data" – stop reading
            print(f"Stopped at CSV error: {e}")
            break

if rows:
    # First row as header
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # Save the fixed CSV
    df.to_csv('data/movies_fixed.csv', index=False)
    print(f"✅ Loaded {len(df)} rows from movies.csv (saved as movies_fixed.csv).")
else:
    print("❌ No rows read.")