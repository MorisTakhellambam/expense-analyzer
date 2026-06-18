import pandas as pd
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.tools import tool

from config import CLEANED_DATA_PATH_WITH_TEXT
from retriever import get_retriever



# load dataframe
df = pd.read_csv(CLEANED_DATA_PATH_WITH_TEXT, parse_dates=['date'])
df['amount']    = df['amount'].astype(float)
df['category']  = df['category'].str.strip().str.title()



# tool 1 - semantic search
@tool
def expense_search(
    query: str,
    category: Optional[str] = None,
    date: Optional[str] = None,
    day: Optional[str] = None,
    month: Optional[str] = None
) -> str:
    """
    Semantically searches expense history and returns matching records.
    Use for open-ended questions like 'what did I spend on dining out?'
    or 'show me travel expenses in March'.

    Args:
        query    : natural language search query
        category : optional, e.g. 'Food'
        date     : optional exact date as 'DD/MM/YYYY', e.g. '01/01/2026'
        day      : optional day of week, e.g. 'Saturday'
        month    : optional month name, e.g. 'March'
    """
    retriever = get_retriever(
                    category=category,
                    date=date,
                    day=day,
                    month=month,
                    k=15
                )
    docs = retriever.invoke(query)

    if not docs:
        return "No matching expenses found."
    
    results = "\n".join(f"- {doc.page_content}" for doc in docs)
    return f"Found {len(docs)} matching expenses: \n{results}"



# tool 2 - calculator
@tool
def expense_calculator(
    operation: str,
    category: Optional[str] = None,
    date: Optional[str] = None,
    day: Optional[str] = None,
    month: Optional[str] = None
) -> str:
    """
    Computes totals, averages, counts, max or min over expenses.
    Use for questions like:
      - 'total spend on 2025-01-01'
      - 'average Food spend in March'
      - 'how much did I spend on Saturdays?'
      - 'max expense in February'

    Args:
        operation : 'total' | 'average' | 'count' | 'max' | 'min'
        category  : optional, e.g. 'Food'
        date      : optional exact date as 'DD/MM/YYYY', e.g. '01/01/2026'
        day       : optional day of week, e.g. 'Saturday'
        month     : optional month name, e.g. 'January'
    """
    filtered = df.copy()

    if date:     filtered = filtered[filtered["date"] == date]
    if month:    filtered = filtered[filtered["month"].str.lower() == month.lower()]
    if day:      filtered = filtered[filtered["day"].str.lower() == day.lower()]
    if category: filtered = filtered[filtered["category"].str.lower() == category.lower()]

    if filtered.empty:
        return "No expenses found for the given filters."
    
    scope = " ".join(filter(None,[
        f"on {date}"                if date     else None,
        f"on {day.capitalize()}s"   if day      else None,
        f"in {month}"               if month    else None,
        f"under '{category}'"       if category else None,
    ])) or "across all records"

    op = operation.lower()
    if op == "total":
        return f"Total {scope}: {filtered['amount'].sum():,.2f} ({len(filtered)} transactions)."
    elif op == "average":
        return f"Average {scope}: {filtered['amount'].mean():,.2f} ({len(filtered)} transactions)."
    elif op == "count":
        return f"Transactions {scope}: {len(filtered)}."
    elif op == "max":
        row = filtered.loc[filtered['amount'].idxmax()]
        return f"Highest {scope}: {row['amount']:,.2f} - {row['description']} ({row['date']})."
    elif op == "min":
        row = filtered.loc[filtered["amount"].idxmin()]
        return f"Lowest {scope}: ₹{row['amount']:,.2f} — {row['description']} ({row['date']})."
    else:
        return f"Unknown operation '{op}'. Use: total, average, count, max, min."



# tool 3 - comparator
@tool
def expense_comparator(
    compare_by: str,
    value_a: str,
    value_b: str,
    category: Optional[str] = None,
    month: Optional[str] = None
) -> str:
    """
    Compares spending between two groups.
    Use for questions like:
      - 'did I spend more in January or February?'
      - 'compare Food vs Transport spending'
      - 'do I spend more on Mondays or Saturdays?'

    Args:
        compare_by : 'month' | 'category' | 'day'
        value_a    : first group,  e.g. 'January', 'Food', or 'Monday'
        value_b    : second group, e.g. 'February', 'Transport', or 'Saturday'
        category   : optional extra filter (when compare_by='month' or 'day')
        month      : optional extra filter (when compare_by='day')
    """

    valid = {"month", "category", "day"}
    if compare_by not in valid:
        return f"compare_by must be one of: {", ".join(valid)}."
    
    col = compare_by        # col name match valid

    def get_total(val: str):
        subset = df[df[col].str.lower() == val.lower()]
        if category and compare_by in ("month", "day"):
            subset = subset[subset["category"].str.lower() == category.lower()]
        if month and compare_by == "day":
            subset = subset[subset["month"].str.lower() == month.lower()]
        return subset["amount"].sum(), len(subset)

    total_a, count_a = get_total(value_a)
    total_b, count_b = get_total(value_b)

    if total_a == 0 and total_b == 0:
        return f"No data found for '{value_a}' or '{value_b}'."
    
    winner = value_a if total_a >= total_b else value_b
    diff = abs(total_a - total_b)

    extra_filters = " ".join(filter(None, [
        f"category='{category}'" if category else None,
        f"month='{month}'"       if month    else None,
    ]))
    extra = f" (filtered to {extra_filters})" if extra_filters else ""

    return (
        f"Comparison by {compare_by}{extra}:\n"
        f"- {value_a}: ₹{total_a:,.2f} ({count_a} transactions)\n"
        f"- {value_b}: ₹{total_b:,.2f} ({count_b} transactions)\n"
        f"- {winner} had higher spending by ₹{diff:,.2f}."
    )