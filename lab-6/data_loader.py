"""
Data Loader Module for Book Recommendation Engine
CS 5130 - Lab 6

This module demonstrates efficient data loading techniques covered in Week 12:
- Selective column loading to reduce memory usage
- Optimal dtype specification for memory efficiency
- Data preprocessing and cleaning
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import ast


class BookDataLoader:
    """
    Efficiently loads and preprocesses the Goodreads book dataset.

    Demonstrates Week 12 concepts:
    - Memory-efficient data loading with usecols
    - dtype optimization
    - Chunked processing (optional for very large datasets)
    """

    # Only keep columns the recommendation models actually need
    COLUMNS_TO_LOAD = [
        'book_id',
        'book_title',
        'author',
        'genres',
        'num_ratings',
        'num_reviews',
        'average_rating',
        'num_pages'
    ]

    # Light-weight dtypes keep memory usage down on the large Goodreads dump
    DTYPE_SPECIFICATION = {
        'book_id': 'int32',  # IDs don't need int64
        'book_title': 'string',  # Use pandas string type
        'author': 'string',
        'genres': 'string',
        'num_ratings': 'int32',  # Rating counts fit in int32
        'num_reviews': 'int32',
        'average_rating': 'float32',  # float32 is sufficient for ratings
        'num_pages': 'string'  # Will need parsing (stored as list)
    }

    def __init__(self, filepath: str):
        """
        Initialize the data loader.

        Args:
            filepath: Path to the CSV file
        """
        self.filepath = filepath
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset with optimal memory usage.

        Returns:
            DataFrame with loaded data
        """
        self.data = pd.read_csv(
            self.filepath,
            usecols=self.COLUMNS_TO_LOAD,
            dtype=self.DTYPE_SPECIFICATION,
        )

        # Ensure book_id remains a column even if the source file has an unnamed index
        if 'book_id' not in self.data.columns:
            # Some CSVs store book_id as the index; reset to make it a column again
            if self.data.index.name == 'book_id':
                self.data = self.data.reset_index()
            else:
                self.data = self.data.reset_index().rename(columns={'index': 'book_id'})

        # Enforce correct dtype after any index reset
        self.data['book_id'] = self.data['book_id'].astype(self.DTYPE_SPECIFICATION['book_id'])

        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the loaded data.

        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.data['genres'] = self.data['genres'].apply(self._parse_genres)

        self.data['num_pages'] = self.data['num_pages'].apply(self._parse_pages)

        self.data = self.data.dropna(subset=['book_title', 'author', 'average_rating'])

        # Fill missing genres with empty list
        self.data['genres'] = self.data['genres'].apply(lambda x: x if isinstance(x, list) else [])

        # Fill missing pages with median
        median_pages = self.data['num_pages'].median()
        self.data['num_pages'].fillna(median_pages, inplace=True)

        self.data['popularity_score'] = self._calculate_popularity()

        return self.data

    def _parse_genres(self, genre_str: str) -> List[str]:
        """
        Parse genres from string representation to list.

        Args:
            genre_str: String representation of genre list

        Returns:
            List of genres
        """
        if pd.isna(genre_str):
            return []

        try:
            genres = ast.literal_eval(genre_str)
            return genres if isinstance(genres, list) else []
        except (ValueError, SyntaxError, TypeError):
            return []

    def _parse_pages(self, pages_str: str) -> float:
        """
        Parse number of pages from string representation.

        Args:
            pages_str: String representation of pages list

        Returns:
            Number of pages as float (or np.nan if invalid)
        """
        if pd.isna(pages_str):
            return np.nan

        try:
            pages_list = ast.literal_eval(pages_str)
            if isinstance(pages_list, list) and len(pages_list) > 0:
                return float(pages_list[0])
        except (ValueError, SyntaxError, TypeError):
            pass
        return np.nan

    def _calculate_popularity(self) -> pd.Series:
        """
        Calculate a popularity score for each book.

        Week 10 concept: Use vectorized operations for efficiency!

        Returns:
            Series of popularity scores
        """
        def _normalize(series: pd.Series) -> pd.Series:
            max_value = series.max()
            if pd.isna(max_value) or max_value == 0:
                return pd.Series(0, index=series.index, dtype='float32')
            return (series / max_value).astype('float32')

        # Ratings give a sense of overall engagement, reviews capture depth of engagement
        normalized_ratings = _normalize(np.log1p(self.data['num_ratings']))
        normalized_reviews = _normalize(np.log1p(self.data['num_reviews']))

        engagement = (0.7 * normalized_ratings + 0.3 * normalized_reviews).astype('float32')
        sentiment = (self.data['average_rating'] / 5.0).astype('float32')

        popularity = engagement * sentiment

        return popularity.astype('float32')

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Report memory usage statistics.

        This demonstrates the impact of dtype optimization (Week 12 concept)

        Returns:
            Dictionary with memory usage information
        """
        if self.data is None:
            return {"error": "No data loaded"}

        memory_usage = self.data.memory_usage(deep=True)

        return {
            "total_memory_mb": memory_usage.sum() / 1024 ** 2,
            "per_column_mb": (memory_usage / 1024 ** 2).to_dict(),
            "shape": self.data.shape
        }

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Convenience method to load and preprocess in one call.

        Returns:
            Preprocessed DataFrame ready for recommendation engine
        """
        self.load_data()
        return self.preprocess_data()


# Example usage and testing
if __name__ == "__main__":
    # Test the data loader
    loader = BookDataLoader("data/Book_Details.csv")

    print("Loading data...")
    df = loader.load_and_preprocess()

    print("\n=== Data Info ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    print("\n=== Memory Usage ===")
    memory_info = loader.get_memory_usage()
    print(f"Total Memory: {memory_info['total_memory_mb']:.2f} MB")

    print("\n=== Sample Data ===")
    print(df.head())

    print("\n=== Data Types ===")
    print(df.dtypes)
