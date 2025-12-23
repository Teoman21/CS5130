# Lab 6: Book Recommendation Engine

**CS 5130: Applied Programming and Data Processing for AI**  
**Khoury College of Computer Sciences, Northeastern University**

## 📚 Overview

Build a functional book recommendation engine using real Goodreads data! This lab integrates concepts from Weeks 6-12 (Module 2):
- Efficient data loading and processing
- Object-oriented design patterns
- Algorithmic optimization
- Clean, maintainable code

## 🎯 Learning Objectives

You will practice:
- **Week 12**: Memory-efficient data loading with selective columns and dtype optimization
- **Week 10**: Vectorized operations for performance
- **Week 8**: Strategy pattern for extensible design
- **Week 7**: Functional programming concepts
- **Week 9**: Writing clean, modular code

## 📥 Setup Instructions

### 1. Download the Dataset

Get the Goodreads Books Dataset (May 2024) from Kaggle:
- Link: https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024
- You'll need a free Kaggle account
- Download the CSV file and place it in your project directory
- Rename it to `goodreads_books_2024.csv` (or update the filename in the code)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- pandas (data processing)
- numpy (numerical operations)
- scikit-learn (machine learning utilities)

### 3. Project Structure

```
lab6/
├── data_loader.py          # Data loading and preprocessing
├── book_recommender.py     # Recommendation engine
├── demo.py                 # Demonstration script
├── requirements.txt        # Dependencies
├── README.md              # This file
└── goodreads_books_2024.csv  # Dataset (you download this)
```

## 🚀 Getting Started

### Quick Test

Run the demo script to see if everything works:

```bash
python demo.py
```

This will:
1. Load and preprocess the data
2. Build the recommendation engine
3. Demonstrate all three recommendation strategies
4. Show performance metrics

### Development Workflow

1. **Start with `data_loader.py`**
   - Complete the TODOs for efficient data loading
   - Test with `python data_loader.py`
   - Verify memory usage and data types

2. **Move to `book_recommender.py`**
   - Implement content-based recommendations first
   - Then add popularity and hybrid strategies
   - Test individual components

3. **Create your demo**
   - Modify `demo.py` to showcase your implementation
   - Test with different books
   - Compare strategies

4. **Write your report**
   - Document your design decisions
   - Include performance measurements
   - Compare recommendation strategies

## 💡 Implementation Tips

### Data Loading
- Use `pd.read_csv()` with `usecols` parameter to load only necessary columns
- Specify optimal dtypes to reduce memory usage
- Use `ast.literal_eval()` to parse string representations of lists (NOT `eval()`!)

### Content-Based Filtering
- Use `MultiLabelBinarizer` for one-hot encoding genres
- Normalize numerical features to 0-1 scale
- Use `sklearn.metrics.pairwise.cosine_similarity` for efficiency
- Remember to exclude the input book from recommendations!

### Performance
- Pre-compute the similarity matrix (one-time cost)
- Use vectorized NumPy operations instead of loops
- Time your operations to measure impact of optimizations

### Testing Your Recommendations
Try these books to see if recommendations make sense:
- "Harry Potter and the Half-Blood Prince" → Should get other fantasy/YA books
- "1984" → Should get other dystopian/classic fiction
- "The Great Gatsby" → Should get other American classics

## 📊 What You'll Submit

1. **Code Files** (in a ZIP):
   - `data_loader.py` (completed)
   - `book_recommender.py` (completed)
   - `demo.py` (your demonstration)
   - `requirements.txt`

2. **Written Report** (PDF):
   - Memory optimization analysis
   - Performance measurements
   - Strategy comparison
   - Design decisions
   - (1-2 pages)

**DO NOT include the dataset CSV in your submission** (too large)

## 🔍 Debugging Tips

### Problem: "File not found"
- Make sure `goodreads_books_2024.csv` is in the same directory as your code
- Check the filename matches exactly (case-sensitive!)

### Problem: "Memory error"
- Are you using dtype optimization?
- Are you loading only necessary columns?
- Try testing with fewer rows first: `df.head(1000)`

### Problem: "Recommendations don't make sense"
- Check your similarity computation
- Verify features are normalized
- Make sure you're excluding the input book
- Print similarity scores to debug

### Problem: "Code is too slow"
- Are you using vectorized operations?
- Check if you're using sklearn's cosine_similarity (not loops!)
- Time different parts of your code to find bottlenecks

## 📖 Key Concepts Reference

### Memory Optimization
```python
# Instead of default types:
df = pd.read_csv('file.csv')  # Uses int64, float64

# Use optimized types:
df = pd.read_csv('file.csv', dtype={
    'id': 'int32',      # Saves 50% memory
    'rating': 'float32'  # Saves 50% memory
})
```

### Vectorization Example
```python
# Slow (loop):
for i in range(len(df)):
    result[i] = df['a'][i] * df['b'][i]

# Fast (vectorized):
result = df['a'] * df['b']
```

### Strategy Pattern
```python
# Base class
class Strategy:
    def execute(self): pass

# Concrete strategies
class StrategyA(Strategy):
    def execute(self): return "A"

class StrategyB(Strategy):
    def execute(self): return "B"

# Usage
strategy = StrategyA()
strategy.execute()  # Can swap strategies easily!
```

## 🎓 Learning Resources

- **pandas documentation**: https://pandas.pydata.org/docs/
- **sklearn cosine_similarity**: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- **Strategy Pattern**: https://refactoring.guru/design-patterns/strategy/python/example

## ❓ Getting Help

1. **Check the starter code comments** - They have hints and guidance
2. **Post on the course discussion board** - You can discuss approaches (not share code)
3. **Come to office hours** - We're here to help!
4. **Check the lab writeup** - It has detailed requirements and rubric

## 🎉 Have Fun!

This is your chance to build something practical that you could actually use or show in a portfolio. Test it with books you love, share recommendations with friends, and enjoy seeing your code in action!

Remember: **Your lowest lab grade is dropped**, so use this as an opportunity to experiment and learn without excessive pressure.