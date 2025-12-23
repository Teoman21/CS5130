from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence


class DataProcessor:
    """Base class for all data processing operations."""

    def process(self, data):
        """Process the data and return the result."""
        raise NotImplementedError


class DateParser(DataProcessor):
    """Extract and parse dates from text entries."""

    def __init__(self, date_format: str = "%Y-%m-%d") -> None:
        """
        Initialize the parser.

        Args:
            date_format: String format for parsing dates (default: "%Y-%m-%d")
        """
        self.date_format = date_format

    def process(self, entries: Iterable[str]) -> List[dict]:
        """
        Parse dates from text entries.

        Args:
            entries: Iterable of strings, each potentially containing a date

        Returns:
            List of dictionaries with 'date' (datetime) and 'text' (str) keys
        """
        parsed_entries: List[dict] = []

        for entry in entries:
            if not entry:
                continue

            if not isinstance(entry, str):
                continue

            try:
                date_part, text_part = entry.split(": ", 1)
            except ValueError:
                # Entry does not contain the expected separator, skip it.
                continue

            try:
                parsed_date = datetime.strptime(date_part.strip(), self.date_format)
            except ValueError:
                # Date portion could not be parsed with the provided format.
                continue

            parsed_entries.append({"date": parsed_date, "text": text_part.strip()})

        return parsed_entries


class WeekdayFilter(DataProcessor):
    """Filter entries to keep only specific days of the week."""

    def __init__(self, allowed_days: Sequence[str]):
        """
        Initialize the filter.

        Args:
            allowed_days: List of day names to keep (e.g., ['Monday', 'Friday'])
        """
        self.allowed_days = {day.strip() for day in allowed_days}

    def process(self, entries: Iterable[dict]) -> List[dict]:
        """
        Filter entries by day of week.

        Args:
            entries: Iterable of dictionaries with 'date' and 'text' keys

        Returns:
            List of entries where the date falls on an allowed day
        """
        filtered = []

        for entry in entries:
            date_value = entry.get("date")

            if not hasattr(date_value, "strftime"):
                continue

            day_name = date_value.strftime("%A")
            if day_name in self.allowed_days:
                filtered.append(entry)

        return filtered


class DateFormatter(DataProcessor):
    """Format dates into readable strings."""

    def __init__(self, output_format: str = "%B %d, %Y"):
        """
        Initialize the formatter.

        Args:
            output_format: String format for output dates (default: "%B %d, %Y")
        """
        self.output_format = output_format

    def process(self, entries: Iterable[dict]) -> List[str]:
        """
        Format entries as strings with formatted dates.

        Args:
            entries: Iterable of dictionaries with 'date' and 'text' keys

        Returns:
            List of formatted strings
        """
        formatted_entries: List[str] = []

        for entry in entries:
            date_value = entry.get("date")

            if not hasattr(date_value, "strftime"):
                continue

            text_value = entry.get("text", "")
            formatted_date = date_value.strftime(self.output_format)
            formatted_entries.append(f"{formatted_date}: {text_value}")

        return formatted_entries


class ProcessingPipeline:
    """Chain multiple processors together."""

    def __init__(self, processors: Sequence[DataProcessor]):
        self.processors = processors

    def process(self, data):
        """Run data through all processors in sequence."""
        result = data
        for processor in self.processors:
            result = processor.process(result)
        return result



# Test 1: DateParser basic functionality
parser = DateParser(date_format="%Y-%m-%d")
entries = ["2024-10-15: Event 1", "2024-10-16: Event 2"]
result = parser.process(entries)
print(f"Parsed {len(result)} entries")  # Should be 2

# Test 2: DateParser with invalid entries
parser = DateParser(date_format="%Y-%m-%d")
entries = ["2024-10-15: Valid", "Not a date", "2024-10-16: Also valid"]
result = parser.process(entries)
print(f"Parsed {len(result)} entries")  # Should be 2 (skips invalid)

# Test 3: WeekdayFilter
filter = WeekdayFilter(allowed_days=['Monday'])
entries = [
    {'date': datetime(2024, 10, 14), 'text': 'Monday'},
    {'date': datetime(2024, 10, 15), 'text': 'Tuesday'}
]
result = filter.process(entries)
print(f"Filtered to {len(result)} entries")  # Should be 1

# Test 4: DateFormatter
formatter = DateFormatter(output_format="%B %d")
entries = [{'date': datetime(2024, 10, 15), 'text': 'Test'}]
result = formatter.process(entries)
print(result[0])  # Should be "October 15: Test"

# Test 5: Full pipeline
pipeline = ProcessingPipeline([
    DateParser(date_format="%Y-%m-%d"),
    WeekdayFilter(allowed_days=['Monday', 'Wednesday']),
    DateFormatter(output_format="%A, %B %d")
])
logs = [
    "2024-10-14: Monday event",
    "2024-10-15: Tuesday event",
    "2024-10-16: Wednesday event"
]
result = pipeline.process(logs)
print(result)  # Should have 2 formatted entries (Monday and Wednesday)