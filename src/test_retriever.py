
from retriever import query

# Pure semantic — no filters
print("=== Dining related ===")
for r in query("restaurants and eating out"):
    print(" •", r)

print("\n=== Transport related ===")
for r in query("transport and travel"):
    print(" •", r)

# Filter by month only
print("\n=== May related ===")
for r in query("all expenses", month="May"):
    print(" •", r)

# Filter by category only
print("\n=== Food category ===")
for r in query("food and drinks", category="Food"):
    print(" •", r)

# Both filters combined
print("\n=== Food in 21-01-2026 ===")
for r in query("meals", category="Food", date="21/01/2026"):
    print(" •", r)