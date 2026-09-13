# =========================================================
# data_cleaning.py
# Owns the DB connection. Loads raw tables from MySQL, cleans them,
# then recreates the cleaned tables (with their PK/FK constraints)
# inside a fresh in-memory SQLite engine.
#
# Exposes: `engine` — the cleaned, constrained in-memory engine.
# rag_common.py imports this `engine` directly; nothing else changes.
# =========================================================
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text

# ---------------------------------------------------------
# 1. Connect to MySQL and load raw tables into pandas
# ---------------------------------------------------------
mysql_engine = create_engine("mysql+pymysql://root:#Shorya1375@localhost/rag_evaluation")
mysql_inspector = inspect(mysql_engine)

table_names = mysql_inspector.get_table_names()
tables = {name: pd.read_sql(f"SELECT * FROM {name}", mysql_engine) for name in table_names}
print("Loaded raw tables:", list(tables.keys()))


# ---------------------------------------------------------
# 2. Clean the data (missing values + outlier detection)
# ---------------------------------------------------------
for table_name, df in tables.items():
    numeric_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(include="object").columns

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna(df[column].median())

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].fillna("Unknown")

# Final missing-value check
for table_name, df in tables.items():
    total_missing = df.isnull().sum().sum()
    print(f"{table_name}: {total_missing} missing values")

# Outlier detection (IQR) — reporting only, not removing
for table_name, df in tables.items():
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"{column}: {len(outliers)} potential outliers")

print("\nData cleaning complete.")


# ---------------------------------------------------------
# 3. Recreate the cleaned tables in a fresh in-memory engine,
#    carrying over the PK/FK constraints that already exist on
#    the MySQL tables (so downstream schema inspection still
#    sees real relationships, not just plain columns).
# ---------------------------------------------------------
def sqlite_type_for(sa_type):
    type_str = str(sa_type).upper()
    if "INT" in type_str:
        return "INTEGER"
    if any(t in type_str for t in ("FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")):
        return "REAL"
    return "TEXT"


def build_create_table_ddl(table_name, mysql_inspector):
    columns = mysql_inspector.get_columns(table_name)
    pk_info = mysql_inspector.get_pk_constraint(table_name)
    pk_cols = pk_info.get("constrained_columns", [])
    fks = mysql_inspector.get_foreign_keys(table_name)

    column_defs = [f'"{col["name"]}" {sqlite_type_for(col["type"])}' for col in columns]

    if pk_cols:
        column_defs.append(f'PRIMARY KEY ({", ".join(f\'"{c}"\' for c in pk_cols)})')

    for fk in fks:
        child_col = fk["constrained_columns"][0]
        parent_table = fk["referred_table"]
        parent_col = fk["referred_columns"][0]
        column_defs.append(
            f'FOREIGN KEY ("{child_col}") REFERENCES "{parent_table}"("{parent_col}")'
        )

    ddl = f'CREATE TABLE "{table_name}" (\n  ' + ",\n  ".join(column_defs) + "\n);"
    return ddl


engine = create_engine("sqlite:///:memory:")

with engine.begin() as conn:
    for table_name in table_names:
        ddl = build_create_table_ddl(table_name, mysql_inspector)
        conn.execute(text(ddl))

for table_name, df in tables.items():
    df.to_sql(table_name, con=engine, index=False, if_exists="append")

print("\nCleaned, constrained tables loaded into in-memory engine:", list(tables.keys()))

# NOTE: SQLite does not validate FOREIGN KEY / PRIMARY KEY at CREATE TABLE
# time regardless of table creation order, so the loop above is safe even
# though tables are created in whatever order MySQL happens to list them.
