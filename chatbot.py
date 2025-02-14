import os
import requests
from duckduckgo_search import DDGS
from playwright.sync_api import sync_playwright
import google.generativeai as genai
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API keys (Replace with your actual keys)
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "api")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "api")

# Validate API keys are present
if not SERPER_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing required API keys. Please set SERPER_API_KEY and GEMINI_API_KEY environment variables.")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_google(query: str) -> List[str]:
    """Search Google using Serper API with retry mechanism."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        return [result["link"] for result in results.get("organic", [])][:5]  # Get top 5 results
    except requests.exceptions.RequestException as e:
        print(f"Google Search Error: {e}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_duckduckgo(query: str, max_results: int = 5) -> List[str]:
    """Search DuckDuckGo with retry mechanism."""
    try:
        with DDGS() as ddgs:
            return [result["href"] for result in ddgs.text(query, max_results=max_results)]
    except Exception as e:
        print(f"DuckDuckGo Search Error: {e}")
        return []

def scrape_with_playwright(url: str) -> Optional[Dict[str, str]]:
    """Scrape webpage content using Playwright with improved error handling."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            # Set longer timeout and wait for network idle
            page.goto(url, timeout=30000, wait_until='networkidle')
            page.wait_for_load_state('load')
            
            # Wait for content to load
            page.wait_for_selector('body', timeout=10000)
            html_content = page.content()
            
            # Get page title
            title = page.title()
            
            context.close()
            browser.close()
        
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer']):
            element.decompose()
            
        # Get text from multiple elements
        text_elements = []
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'section']):
            text = element.get_text(strip=True)
            if text and len(text) > 50:  # Filter out very short snippets
                text_elements.append(text)
        
        content = "\n\n".join(text_elements)
        
        return {
            "url": url,
            "title": title,
            "content": content[:4000]  # Limit content per URL
        }
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def scrape_multiple_urls(urls: List[str], max_urls: int = 8) -> List[Dict[str, str]]:
    """Scrape multiple URLs in parallel."""
    scraped_data = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_url = {executor.submit(scrape_with_playwright, url): url 
                        for url in urls[:max_urls]}
        
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                scraped_data.append(result)
    
    return scraped_data

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def summarize_with_gemini(content: str) -> str:
    """Summarize content using Gemini AI with retry mechanism."""
    if not content:
        return "No content to summarize."
    
    try:
        prompt = (
            "Please provide a comprehensive summary of the following content from multiple sources, "
            "highlighting the main points and key information. Organize the summary in a clear structure:\n\n"
            f"{content}"
        )
        response = model.generate_content(prompt)
        return response.text if response else "Failed to generate summary."
    except Exception as e:
        print(f"Gemini AI Error: {e}")
        return "Failed to generate summary due to an error."

def main():
    """Main execution with multi-URL scraping and summarization."""
    try:
        user_query = input("Enter your search query: ")
        print("\nSearching multiple sources...")

        # Search using both engines
        google_results = search_google(user_query)
        duckduckgo_results = search_duckduckgo(user_query)

        # Combine and deduplicate results
        all_urls = list(dict.fromkeys(google_results + duckduckgo_results))

        if not all_urls:
            print("No search results found.")
            return

        print(f"\nFound {len(all_urls)} unique URLs. Starting content scraping...")
        scraped_results = scrape_multiple_urls(all_urls)

        if not scraped_results:
            print("Failed to scrape content from any of the URLs.")
            return

        print(f"\nSuccessfully scraped {len(scraped_results)} pages:")
        for result in scraped_results:
            print(f"- {result['title']} ({result['url']})")

        # Combine content from all sources
        combined_content = "\n\nSOURCE SEPARATION\n\n".join(
            f"From {result['title']}:\n{result['content']}"
            for result in scraped_results
        )

        print("\nGenerating comprehensive summary...")
        summary = summarize_with_gemini(combined_content)
        
        print("\n=== AI Summary of Multiple Sources ===\n")
        print(summary)

        print("\n=== Sources Used ===")
        for result in scraped_results:
            print(f"- {result['url']}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
