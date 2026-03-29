# indexes.py
# Sirf ek baar chalao — phir delete kar sakte ho
from config import cows_col

cows_col.create_index("cow_id", unique=True)
cows_col.create_index("owner_phone")
cows_col.create_index("district")
cows_col.create_index("state")
cows_col.create_index("stolen")

print("✓ Saare indexes ban gaye!")