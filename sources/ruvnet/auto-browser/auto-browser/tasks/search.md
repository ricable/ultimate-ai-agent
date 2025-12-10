# Search and Extract Task

## Description
Search for a specific term on a website and extract relevant information.

## Steps
1. Navigate to the search page
2. Find the search input field
3. Enter the search term
4. Submit the search
5. Wait for results to load
6. Extract relevant information from results
7. Verify data was found

## Parameters
- search_term: The term to search for (required)

## Credentials
None

## Required Parameters
- search_term

## Expected Output
Extracted data should include:
- Search result titles
- Relevant prices or metrics
- Timestamps or dates
- Any associated metadata

## Selectors
- Search field: input[type='search'], .search-input
- Submit button: button[type='submit'], .search-submit
- Results container: .search-results, .results-list
- Result items: .result-item, .search-result
- Price elements: .price, .value
- Date elements: .date, .timestamp
