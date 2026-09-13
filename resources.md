# DataSentry Learning Resources

## Step 1 — Project Structure

### Python Packages & `__init__.py`
- https://docs.python.org/3/tutorial/modules.html#packages — Official docs on modules and packages (sections 6.1, 6.4)
- YouTube: "Corey Schafer Python modules and packages" (~15 min)

### `src/` Layout
- https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/ — Why use src/ layout

### `.gitignore`
- YouTube: "Fireship .gitignore in 100 seconds"
- YouTube: "Fireship Git it 100 seconds" (if new to Git)

### Separation of Concerns
- YouTube: "Fireship single responsibility principle" (100 seconds)

---

## Step 2 — `pyproject.toml`

- YouTube: "ArjanCodes pyproject.toml" (~15 min)
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ — Official guide

---

## Step 3 — `config.py` (Dataclasses)

- YouTube: "mCoding Python dataclasses" (~10 min)
- Google: "python dataclass mutable default" — why lists need `field(default_factory=...)`

---

## Step 4 — `ingestion.py`

### pandas `read_csv` and `na_values`
- https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html — look for `na_values` parameter

### File Encoding
- YouTube: "Corey Schafer Python file encoding"
- https://docs.python.org/3/howto/unicode.html — intro section

---

## Step 5 — `test_ingestion.py` (pytest)

- YouTube: "mCoding pytest" (~10 min) — covers test functions, assert, fixtures
- https://docs.pytest.org/en/stable/how-to/tmp_path.html — `tmp_path` fixture for temporary test files
- https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions — `pytest.raises` for testing errors

### Python Concepts Used
- `pathlib.Path` and `/` operator for joining paths
- `try/except` (Python's version of Java's try/catch)
- `raise` (Python's version of Java's `throw`)
- f-strings: `f"File not found: {path}"` (like Java's `String.format()`)

---

## Big Picture — ML & Data Concepts

- YouTube: "StatQuest Machine Learning fundamentals" — first 2-3 videos for vocabulary (features, target, training data)
- YouTube: "data preprocessing for machine learning" — any 10-15 min video on dirty data
