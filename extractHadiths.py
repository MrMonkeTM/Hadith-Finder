import asyncio
import json
from playwright.async_api import async_playwright
from itertools import zip_longest

async def main():
    all_hadiths = []  # Step 1: create empty list

    async with async_playwright() as p:
        books = {
            "bukhari": 97,
            "muslim": 56,
            "nasai": 51,
            "abudawud": 43,
            "tirmidhi": 49,
            "ibnmajah": 37
        }
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for book, chapter_count in books.items():
            for chapter_num in range(1, chapter_count + 1):
                await page.route("**/*", lambda route: route.abort()
                    if route.request.resource_type in ["image", "stylesheet", "font", "media", "other", "script"]
                    else route.continue_())

                url = f"https://sunnah.com/{book}/{chapter_num}"
                print(f"Scraping: {url}")
                try:
                    await page.goto(url, wait_until="domcontentloaded")
                    await page.wait_for_selector("div.text_details")

                    titles = await page.locator("div.hadith_narrated").all_inner_texts()
                    divs = await page.locator("div.text_details").all_inner_texts()
                    references = await page.locator("table.hadith_reference a:last-child").all_inner_texts()

                    for title, div, ref in zip_longest(titles, divs, references, fillvalue=""):
                        hadith = {
                            "book": book,
                            "reference": ref.strip(),       # ✅ actual ref like Muslim 226a
                            "title": title.strip(),
                            "text": div.strip()
                        }
                        all_hadiths.append(hadith)


                except Exception as e:
                    print(f"Error at chapter {chapter_num}: {e}")
                    continue

        await browser.close()

    with open("all_hadiths.json", "w", encoding="utf-8") as f:
        json.dump(all_hadiths, f, ensure_ascii=False, indent=2)
        print("✅ All hadiths saved to all_hadiths.json")

asyncio.run(main())


# change the linkes to follow sunnah.com/{book}/{chapter_num} format
# ^^^ and add a criteria to .json to include the reference number (this helps account for 226a and 226b, etc.)