import os
import requests
from ddgs import DDGS
from playwright.sync_api import sync_playwright
import google.generativeai as genai
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load API keys (Replace with your actual keys)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API keys are present
if not SERPER_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing required API keys. Please set SERPER_API_KEY and GEMINI_API_KEY environment variables.")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

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
def summarize_with_gemini(content: str, is_individual: bool = False, source_title: str = "") -> str:
    """Summarize content using Gemini AI with retry mechanism."""
    if not content:
        return "No content to summarize."
    
    try:
        if is_individual:
            prompt = (
                f"Please provide a concise and focused summary of the following content from '{source_title}'. "
                "Extract the key points, main ideas, and important information in a clear and structured way:\n\n"
                f"{content}"
            )
        else:
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

def generate_individual_summaries(scraped_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Generate individual summaries for each scraped result."""
    print("Generating individual summaries for each source...")
    
    def summarize_single_source(result):
        summary = summarize_with_gemini(result["content"], is_individual=True, source_title=result["title"])
        return {
            **result,
            "individual_summary": summary
        }
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_result = {executor.submit(summarize_single_source, result): result 
                          for result in scraped_results}
        
        summarized_results = []
        for future in as_completed(future_to_result):
            summarized_results.append(future.result())
    
    return summarized_results

def save_json_output(data: dict, query: str) -> str:
    """Save JSON output to a file and return filename."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Clean query for filename
    clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_query = clean_query.replace(' ', '_')[:50]  # Limit filename length
    
    filename = f"search_results_{clean_query}_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"Failed to save JSON file: {e}")
        return ""

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
            result = {
                "status": "error",
                "message": "No search results found",
                "query": user_query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sources": [],
                "summary": ""
            }
            print(json.dumps(result, indent=2))
            return

        print(f"\nFound {len(all_urls)} unique URLs. Starting content scraping...")
        scraped_results = scrape_multiple_urls(all_urls)

        if not scraped_results:
            result = {
                "status": "error",
                "message": "Failed to scrape content from any of the URLs",
                "query": user_query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sources": [{"url": url, "scraped": False, "error": "Failed to scrape"} for url in all_urls],
                "summary": ""
            }
            print(json.dumps(result, indent=2))
            return

        print(f"\nSuccessfully scraped {len(scraped_results)} pages")

        # Generate individual summaries for each source
        scraped_results_with_summaries = generate_individual_summaries(scraped_results)

        # Combine content from all sources
        combined_content = "\n\nSOURCE SEPARATION\n\n".join(
            f"From {result['title']}:\n{result['content']}"
            for result in scraped_results_with_summaries
        )

        print("\nGenerating comprehensive summary...")
        overall_summary = summarize_with_gemini(combined_content)
        
        # Create structured JSON output
        structured_result = {
            "status": "success",
            "query": user_query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_sources_found": len(all_urls),
            "successfully_scraped": len(scraped_results_with_summaries),
            "sources": [
                {
                    "url": result["url"],
                    "title": result["title"],
                    "content_preview": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "individual_summary": result["individual_summary"],
                    "scraped": True
                }
                for result in scraped_results_with_summaries
            ],
            "failed_sources": [
                {"url": url, "scraped": False}
                for url in all_urls
                if url not in [r["url"] for r in scraped_results_with_summaries]
            ],
            "overall_summary": overall_summary,
            "metadata": {
                "google_results_count": len(google_results),
                "duckduckgo_results_count": len(duckduckgo_results),
                "total_unique_urls": len(all_urls),
                "processing_time": "Real-time processing completed"
            }
        }
        
        print("\n" + "="*50)
        print("STRUCTURED JSON OUTPUT")
        print("="*50)
        print(json.dumps(structured_result, indent=2, ensure_ascii=False))
        
        # Ask user if they want to save the output
        save_option = input("\nWould you like to save this result to a JSON file? (y/n): ").lower().strip()
        if save_option in ['y', 'yes']:
            filename = save_json_output(structured_result, user_query)
            if filename:
                print(f"Results saved to: {filename}")
            else:
                print("Failed to save file.")

    except KeyboardInterrupt:
        error_result = {
            "status": "cancelled",
            "message": "Operation cancelled by user",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print(json.dumps(error_result, indent=2))
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        print(json.dumps(error_result, indent=2))

if __name__ == "__main__":
    main()
