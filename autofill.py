"""
Google Form Auto-Fill Bot  —  Fixed & Robust Version
Form : Competitive Benchmarking of Portable Audio Devices
URL  : https://forms.gle/LSQ8Bkyo3gr7KhFKA

5-page form structure
─────────────────────────────────────────────────────────
Page 1  Demographics           5 radio questions
Page 2  Product Selection      1 checkbox  +  3 radio questions
Page 3  Influencer Attributes  6 Likert scale (1–5) questions
Page 4  Campaign Realism       6 Likert scale (1–5) questions
Page 5  Purchase Intention     4 Likert scale  +  1 radio question
─────────────────────────────────────────────────────────

Requirements:
    pip install selenium webdriver-manager

Usage:
    python3 autofill.py                       # 1 submission, default answers
    python3 autofill.py --times 5 --random    # 5 random submissions
    python3 autofill.py --headless            # invisible browser
"""

import time
import random
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ─── Form URL ────────────────────────────────────────────────────────────────
FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdRE1Fv6k2ffzkyqAVaNodwAj4KAAgjU9yoAC0aE6tHV5p0wA/viewform"
)

# ─── Product options (used on pages 2 & 5) ───────────────────────────────────
PRODUCTS = ["Boat Nirvana Ivy", "OnePlus Nord Buds 3 Pro", "Realme Buds Air 7"]

# ─── Page 1 – Demographics ───────────────────────────────────────────────────
DEFAULT_P1 = {
    "age":        "22-25",
    "gender":     "Male",
    "occupation": "Student",
    "hours":      "1-2 hours",
    "platform":   "Instagram",
}
RANDOM_P1 = {
    "age":        ["18-21", "22-25", "26-30", "30 and more"],
    "gender":     ["Male", "Female", "Prefer not to say"],
    "occupation": ["Student", "Working Profesional", "Entrepreneur", "Other"],
    "hours":      ["Less than 1 hours", "1-2 hours", "2-4 hours", "More than 4 hours"],
    "platform":   ["Instagram", "Youtube", "Facebook", "X(Twitter)"],
}

# ─── Page 2 – Product Selection ──────────────────────────────────────────────
DEFAULT_CHECKBOXES = ["Good Sound Quality", "Comfortable to daily use"]
ALL_CHECKBOXES = [
    "Good Sound Quality",
    "Features like noise cancellation, connectivity, battery performance etc",
    "Design & appearance",
    "Comfortable to daily use",
    "Reliable brand",
]
DEFAULT_PRODUCT = "OnePlus Nord Buds 3 Pro"

# ─── Likert default (1–5); 4 = "Agree" ───────────────────────────────────────
DEFAULT_SCALE = 4


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def safe_scroll_click(driver, element):
    """Scroll element into view, then click (JS fallback if intercepted)."""
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(0.25)
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def click_radio_by_label(driver, wait, label_text: str):
    """
    Click the first radio option whose visible span matches label_text exactly.
    Safe for Page 1 because every question has unique option labels.
    """
    xpath = f"//span[normalize-space(text())='{label_text}']"
    el = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    safe_scroll_click(driver, el)


def tick_checkboxes(driver, wait, labels: list):
    """Tick each checkbox that matches a label in the given list."""
    for label in labels:
        try:
            xpath = f"//span[normalize-space(text())='{label}']"
            el = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            safe_scroll_click(driver, el)
            print(f"    ✔ [checkbox] {label}")
        except Exception as e:
            print(f"    ⚠  Could not tick '{label}': {e}")


def get_plain_radio_groups(driver):
    """
    Return all div[role='radiogroup'] elements that are NOT Likert scales.
    Likert scales contain children with a numeric data-value attribute (1–5).
    Plain radio groups (product choices, etc.) do not.
    """
    all_groups = driver.find_elements(By.CSS_SELECTOR, "div[role='radiogroup']")
    plain = []
    for g in all_groups:
        cells = g.find_elements(By.CSS_SELECTOR, "[data-value]")
        values = {c.get_attribute("data-value") for c in cells}
        if not values.intersection({"1", "2", "3", "4", "5"}):
            plain.append(g)
    return plain


def answer_plain_radio_groups(driver, choices: list, use_random: bool):
    """
    Answer every plain radio group on the current page in DOM order.

    Parameters
    ----------
    choices   : preferred label for each group (matched by substring, case-insensitive).
                Unused when use_random=True.
    use_random: pick a random option instead of the preferred one.
    """
    groups = get_plain_radio_groups(driver)
    for idx, group in enumerate(groups):
        try:
            # div[role='radio'] is the actual clickable option container in Google Forms
            options = group.find_elements(By.CSS_SELECTOR, "div[role='radio']")
            if not options:
                print(f"    ⚠  No options found in plain radio group #{idx + 1}")
                continue

            if use_random:
                chosen = random.choice(options)
            else:
                preferred = choices[idx] if idx < len(choices) else None
                chosen = options[0]          # safe fallback
                if preferred:
                    for opt in options:
                        if preferred.strip().lower() in opt.text.strip().lower():
                            chosen = opt
                            break

            safe_scroll_click(driver, chosen)
            print(f"    ✔ [radio group {idx + 1}] {chosen.text.strip()}")

        except Exception as e:
            print(f"    ⚠  Failed plain radio group #{idx + 1}: {e}")


def answer_all_linear_scales(driver, use_random: bool) -> int:
    """
    Detect and answer every Likert / linear-scale group on the current page.
    These groups contain child elements with a numeric data-value attribute.

    Returns the number of scale questions answered.
    """
    groups = driver.find_elements(By.CSS_SELECTOR, "div[role='radiogroup']")
    answered = 0
    for group in groups:
        cells = group.find_elements(By.CSS_SELECTOR, "[data-value]")
        values = [c.get_attribute("data-value") for c in cells]

        # Skip groups that are not a 1-5 numeric scale
        if not cells or not any(v in {"1", "2", "3", "4", "5"} for v in values):
            continue

        pick = str(random.randint(1, 5)) if use_random else str(DEFAULT_SCALE)
        target = next(
            (c for c in cells if c.get_attribute("data-value") == pick),
            cells[0],      # fallback to first cell if target value not found
        )
        safe_scroll_click(driver, target)
        print(f"    ✔ [scale] {pick}/5")
        answered += 1
        time.sleep(0.15)

    return answered


def click_next(driver, wait):
    btn = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//span[text()='Next']/..")
    ))
    btn.click()
    time.sleep(2.5)


def click_submit(driver, wait):
    btn = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//span[text()='Submit']/..")
    ))
    btn.click()
    time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-page fill logic
# ══════════════════════════════════════════════════════════════════════════════

def fill_page1(driver, wait, p1):
    """Page 1 – Demographics (5 unique-label radio questions)."""
    print("\n  [Page 1] Demographics")
    for key in ("age", "gender", "occupation", "hours", "platform"):
        label = p1[key]
        click_radio_by_label(driver, wait, label)
        print(f"    ✔ {key}: {label}")
    click_next(driver, wait)


def fill_page2(driver, wait, checkboxes, product, use_random):
    """
    Page 2 – Product Selection Attributes
      • 1 checkbox group  (what you consider when buying)
      • 3 plain radio groups with identical options – product comparisons:
          Q1  Which product did you like most based on the ads?
          Q2  Which product has the best features?
          Q3  Which product offers the best value for money?
    """
    print("\n  [Page 2] Product Selection Attributes")
    tick_checkboxes(driver, wait, checkboxes)

    # All 3 product radio groups get the same choice (or random per group)
    p2_choices = [product, product, product]
    answer_plain_radio_groups(driver, choices=p2_choices, use_random=use_random)
    click_next(driver, wait)


def fill_page3(driver, wait, use_random):
    """Page 3 – Influencer Attributes (6 Likert scales)."""
    print("\n  [Page 3] Influencer Attributes")
    n = answer_all_linear_scales(driver, use_random)
    print(f"    ✔ {n} scale question(s) answered")
    click_next(driver, wait)


def fill_page4(driver, wait, use_random):
    """Page 4 – Social Media Campaign Attributes / Perceived Realism (6 Likert scales)."""
    print("\n  [Page 4] Social Media Campaign Attributes")
    n = answer_all_linear_scales(driver, use_random)
    print(f"    ✔ {n} scale question(s) answered")
    click_next(driver, wait)


def fill_page5(driver, wait, product, use_random):
    """
    Page 5 – Purchase Intention
      • 4 Likert scale questions
      • 1 plain radio  – Which product would you most likely purchase?
    """
    print("\n  [Page 5] Purchase Intention")
    n = answer_all_linear_scales(driver, use_random)
    print(f"    ✔ {n} scale question(s) answered")

    # Final product radio (1 plain radio group on this page)
    answer_plain_radio_groups(driver, choices=[product], use_random=use_random)
    click_submit(driver, wait)


# ══════════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def fill_form_once(driver, wait, p1, checkboxes, product, use_random, num):
    print(f"\n{'═' * 60}")
    print(f"  Submission #{num}")
    print(f"  Demographics : {p1}")
    print(f"  Checkboxes   : {checkboxes}")
    print(f"  Product pick : {product}")
    print(f"{'═' * 60}")

    driver.get(FORM_URL)
    time.sleep(3)   # let the form fully render

    fill_page1(driver, wait, p1)
    fill_page2(driver, wait, checkboxes, product, use_random)
    fill_page3(driver, wait, use_random)
    fill_page4(driver, wait, use_random)
    fill_page5(driver, wait, product, use_random)

    print(f"\n  ✅ Submitted successfully!")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def build_driver(headless: bool):
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    # Hide the webdriver flag so Google Forms doesn't detect automation
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def main():
    parser = argparse.ArgumentParser(
        description="Auto-fill the Portable Audio Devices benchmarking Google Form."
    )
    parser.add_argument("--times",    type=int,  default=1,     help="Number of submissions (default: 1)")
    parser.add_argument("--random",   action="store_true",       help="Randomise all answers")
    parser.add_argument("--headless", action="store_true",       help="Run Chrome in headless mode")
    args = parser.parse_args()

    driver = build_driver(args.headless)
    wait   = WebDriverWait(driver, 15)

    success = 0
    for i in range(1, args.times + 1):
        # Build per-submission answers
        p1 = (
            {k: random.choice(v) for k, v in RANDOM_P1.items()}
            if args.random else DEFAULT_P1.copy()
        )
        checkboxes = (
            random.sample(ALL_CHECKBOXES, random.randint(1, 3))
            if args.random else DEFAULT_CHECKBOXES.copy()
        )
        product = random.choice(PRODUCTS) if args.random else DEFAULT_PRODUCT

        try:
            ok = fill_form_once(driver, wait, p1, checkboxes, product, args.random, i)
            if ok:
                success += 1
        except Exception as e:
            print(f"\n  ❌ Submission #{i} failed: {e}")
            screenshot = f"debug_submission_{i}.png"
            driver.save_screenshot(screenshot)
            print(f"     Screenshot saved → {screenshot}")

        if i < args.times:
            delay = random.uniform(3, 7)
            print(f"\n  ⏳ Waiting {delay:.1f}s before next submission…")
            time.sleep(delay)

    driver.quit()
    print(f"\n{'═' * 60}")
    print(f"  Done!  {success} / {args.times} submissions successful.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()