"""
chatbot.py — Core NLP FAQ Matching Engine

Preprocessing pipeline: Tokenize → Lowercase → Remove punctuation/stopwords → Lemmatize
Matching:  TF-IDF vectorization + Cosine Similarity
"""

import json
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------- #
# Download required NLTK data on first run
# --------------------------------------------------------------------------- #
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] Downloading '{name}'...")
            nltk.download(name, quiet=True)

download_nltk_resources()

# --------------------------------------------------------------------------- #
# NLP helpers
# --------------------------------------------------------------------------- #
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


# --------------------------------------------------------------------------- #
# Small-talk / greeting intent detection
# --------------------------------------------------------------------------- #
_SMALL_TALK = [
    {
        "intent": "greeting",
        "patterns": [
            r"\b(hi|hello|hey|howdy|hiya|greetings|good\s*(morning|afternoon|evening|day))\b"
        ],
        "responses": [
            "Hey there! 👋 How can I help you with your smartphone today?",
            "Hello! 😊 Ask me anything about your phone — battery, camera, apps, and more!",
            "Hi! Great to see you. What smartphone question can I answer for you?",
        ],
    },
    {
        "intent": "thanks",
        "patterns": [
            r"\b(thanks?|thank\s*you|thx|ty|cheers|appreciate|helpful|great\s*answer|perfect)\b"
        ],
        "responses": [
            "You're welcome! 😊 Feel free to ask if you have more questions.",
            "Happy to help! 🙌 Got another smartphone question?",
            "Glad I could help! Let me know if there's anything else. 👍",
        ],
    },
    {
        "intent": "farewell",
        "patterns": [
            r"\b(bye|goodbye|see\s*you|later|take\s*care|ciao|farewell|ttyl|gotta\s*go)\b"
        ],
        "responses": [
            "Goodbye! 👋 Come back anytime you have smartphone questions!",
            "See you later! Take care of that phone! 📱",
            "Bye! Hope I was helpful. Have a great day! 😊",
        ],
    },
    {
        "intent": "identity",
        "patterns": [
            r"\b(who\s*are\s*you|what\s*are\s*you|your\s*name|are\s*you\s*(a\s*)?(bot|robot|ai|human)|introduce\s*yourself)\b"
        ],
        "responses": [
            "I'm SmartBot 🤖 — your smartphone FAQ assistant! I can answer questions about battery life, cameras, storage, repairs, apps, and more.",
        ],
    },
    {
        "intent": "help",
        "patterns": [
            r"\b(help|what\s*can\s*you\s*do|what\s*do\s*you\s*know|topics?|capabilities?|features?)\b"
        ],
        "responses": [
            "I can help you with smartphone topics like:\n🔋 Battery & charging\n📷 Camera tips\n💾 Storage & speed\n🔒 Security & privacy\n📱 Apps & software\n🛠️ Repairs & screen issues\n📡 Connectivity (Wi-Fi, Bluetooth, 5G)\n\nJust ask away!",
        ],
    },
    {
        "intent": "affirmation",
        "patterns": [
            r"^\s*(ok|okay|alright|sure|got\s*it|i\s*see|understood|cool|nice|wow|great|awesome)\s*[!.]*\s*$"
        ],
        "responses": [
            "😊 Is there anything else you'd like to know about your smartphone?",
            "Got it! Feel free to ask any other smartphone questions. 📱",
        ],
    },
]

import random

def detect_small_talk(text: str):
    """
    Check if the input matches any small-talk intent.
    Returns a response string if matched, or None if not.
    """
    lower = text.lower().strip()
    for intent_def in _SMALL_TALK:
        for pattern in intent_def["patterns"]:
            if re.search(pattern, lower):
                return random.choice(intent_def["responses"])
    return None


# --------------------------------------------------------------------------- #
# Phone Catalog
# --------------------------------------------------------------------------- #
class PhoneCatalog:
    """
    Loads phones from phones.json and answers catalog/price queries.
    """

    CATEGORY_LABELS = {
        "budget":     "Budget",
        "mid-range":  "Mid-Range",
        "flagship":   "Flagship",
        "ultra":      "Ultra Premium",
    }

    CATEGORY_RANGES = {
        "budget":     (0,      15000),
        "mid-range":  (15001,  45000),
        "flagship":   (45001,  90000),
        "ultra":      (90001,  999999),
    }

    def __init__(self, phones_path: str = "phones.json"):
        try:
            with open(phones_path, "r", encoding="utf-8") as f:
                self.phones = json.load(f)
            print(f"[Catalog] Loaded {len(self.phones)} phones.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[Catalog] Warning: Could not load phones.json — {e}")
            self.phones = []

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_price(amount: int) -> str:
        """Format number as ₹XX,XXX"""
        return f"₹{amount:,}"

    def _format_phone(self, p: dict, index: int = None) -> str:
        prefix = f"{index}. " if index else "• "
        s = p["specs"]
        return (
            f"{prefix}📱 {p['name']} ({p['brand']})\n"
            f"   💰 Price : {self._format_price(p['price'])}\n"
            f"   🔹 RAM   : {s['ram']}  |  Storage: {s['storage']}\n"
            f"   📷 Camera: {s['camera']}\n"
            f"   🔋 Battery: {s['battery']}  |  Display: {s['display']}"
        )

    def _build_response(self, phones: list, heading: str) -> str:
        if not phones:
            return (
                f"😕 No phones found {heading}.\n"
                "Try asking for a different price range or say 'show all phones'."
            )
        sorted_phones = sorted(phones, key=lambda p: p["price"])
        lines = [f"📋 {heading} — {len(sorted_phones)} phone(s) found:\n"]
        for i, p in enumerate(sorted_phones, 1):
            lines.append(self._format_phone(p, i))
            lines.append("")          # blank line between entries
        lines.append("💬 Ask me about any of these phones for more details!")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Query handlers
    # ------------------------------------------------------------------ #
    def list_all(self) -> str:
        return self._build_response(self.phones, "All Available Phones")

    def filter_by_range(self, low: int, high: int) -> str:
        matched = [p for p in self.phones if low <= p["price"] <= high]
        heading = f"Phones between {self._format_price(low)} and {self._format_price(high)}"
        return self._build_response(matched, heading)

    def filter_under(self, limit: int) -> str:
        matched = [p for p in self.phones if p["price"] <= limit]
        return self._build_response(matched, f"Phones under {self._format_price(limit)}")

    def filter_above(self, limit: int) -> str:
        matched = [p for p in self.phones if p["price"] >= limit]
        return self._build_response(matched, f"Phones above {self._format_price(limit)}")

    def filter_by_category(self, category: str) -> str:
        low, high = self.CATEGORY_RANGES[category]
        matched = [p for p in self.phones if p["category"] == category]
        label = self.CATEGORY_LABELS[category]
        heading = f"{label} Phones ({self._format_price(low)} – {self._format_price(high)})"
        return self._build_response(matched, heading)

    def filter_by_brand(self, brand: str) -> str:
        matched = [p for p in self.phones if p["brand"].lower() == brand.lower()]
        return self._build_response(matched, f"{brand.title()} Phones")


def detect_phone_query(text: str, catalog: "PhoneCatalog"):
    """
    Detect phone catalog / price queries in natural language.
    Returns a response string if matched, or None if not a phone query.

    Handled patterns:
      • "show all phones" / "what phones are available"
      • "phones under 20000" / "below 15k"
      • "phones above 50000" / "over 1 lakh"
      • "phones between 10000 and 30000" / "10k to 30k"
      • "budget phones" / "mid range phones" / "flagship phones"
      • "Apple phones" / "Samsung phones"  (brand filter)
    """
    lower = text.lower().strip()

    # ── Helper: extract numeric value (handles "20000", "20k", "1 lakh") ── #
    def parse_amount(s: str) -> int | None:
        s = s.replace(",", "").strip()
        m = re.match(r"([\d]+\.?\d*)\s*(k|lakh|lac|l)?", s)
        if not m:
            return None
        val = float(m.group(1))
        suffix = (m.group(2) or "").lower()
        if suffix == "k":
            val *= 1000
        elif suffix in ("lakh", "lac", "l"):
            val *= 100000
        return int(val)

    # ── 1. List all phones ─────────────────────────────────────────────── #
    if re.search(
        r"(show\s+all\s+phones?|all\s+(available\s+)?phones?|"
        r"list\s+(of\s+)?phones?|what\s+(phones?|models?|mobiles?|handsets?)\s+are\s+(available|you\s+have)|"
        r"available\s+phones?|phones?\s+available|show\s+phones?)",
        lower,
    ):
        return catalog.list_all()

    # ── 2. Price range: "between X and Y" / "X to Y" ───────────────────── #
    m = re.search(
        r"between\s+([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)\s+(?:and|to|-)\s+([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)",
        lower,
    )
    if m:
        low = parse_amount(m.group(1))
        high = parse_amount(m.group(2))
        if low is not None and high is not None:
            return catalog.filter_by_range(low, high)

    m = re.search(
        r"([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)\s+to\s+([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)",
        lower,
    )
    if m:
        low = parse_amount(m.group(1))
        high = parse_amount(m.group(2))
        if low is not None and high is not None and low < high:
            return catalog.filter_by_range(low, high)

    # ── 3. Under / below / less than / within ──────────────────────────── #
    m = re.search(
        r"(?:under|below|less\s+than|within|upto|up\s+to|cheaper\s+than)\s+([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)",
        lower,
    )
    if m:
        limit = parse_amount(m.group(1))
        if limit is not None:
            return catalog.filter_under(limit)

    # ── 4. Above / over / more than ────────────────────────────────────── #
    m = re.search(
        r"(?:above|over|more\s+than|greater\s+than|costlier\s+than|expensive\s+than)\s+([\d,]+\.?\d*\s*(?:k|lakh|lac|l)?)",
        lower,
    )
    if m:
        limit = parse_amount(m.group(1))
        if limit is not None:
            return catalog.filter_above(limit)

    # ── 5. Category keywords ────────────────────────────────────────────── #
    if re.search(r"\b(budget|cheap|affordable|entry.?level)\b.*(phone|mobile|handset|model|device)", lower):
        return catalog.filter_by_category("budget")
    if re.search(r"\b(mid.?range|middle|moderate)\b.*(phone|mobile|handset|model|device)", lower):
        return catalog.filter_by_category("mid-range")
    if re.search(r"\b(flagship|premium|high.?end)\b.*(phone|mobile|handset|model|device)", lower):
        return catalog.filter_by_category("flagship")
    if re.search(r"\b(ultra|top.?end|best|luxury)\b.*(phone|mobile|handset|model|device)", lower):
        return catalog.filter_by_category("ultra")

    # ── 6. Brand filter ─────────────────────────────────────────────────── #
    brands = ["apple", "samsung", "google", "oneplus", "xiaomi", "realme", "iqoo", "motorola"]
    for brand in brands:
        if re.search(rf"\b{brand}\b.*(phone|mobile|model|device|handset|price|cost)", lower) or \
           re.search(rf"(phone|mobile|model|device|handset|price|cost).*\b{brand}\b", lower):
            # Map "apple" → "Apple", etc.
            brand_display = {"apple": "Apple", "samsung": "Samsung", "google": "Google",
                             "oneplus": "OnePlus", "xiaomi": "Xiaomi", "realme": "Realme",
                             "iqoo": "iQOO", "motorola": "Motorola"}[brand]
            return catalog.filter_by_brand(brand_display)

    return None   # not a phone query


def preprocess(text: str) -> str:
    """
    Normalise a string for TF-IDF vectorisation.

    Steps:
      1. Lowercase
      2. Remove non-alphabetic characters (punctuation, numbers)
      3. Tokenise
      4. Remove stop words
      5. Lemmatise
    Returns a single cleaned string (space-joined tokens).
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)          # keep only letters + spaces
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in _stop_words and tok not in string.punctuation
    ]
    return " ".join(tokens)


# --------------------------------------------------------------------------- #
# FAQ Chatbot class
# --------------------------------------------------------------------------- #
class FAQChatbot:
    """
    Loads FAQs from a JSON file, builds a TF-IDF index over the questions,
    and answers user queries by finding the most similar FAQ.
    """

    SIMILARITY_THRESHOLD = 0.15   # minimum cosine similarity to accept a match
    FALLBACK_RESPONSE = (
        "I'm sorry, I don't have an answer for that. 🤔\n"
        "Try asking about smartphone tips, or say 'show all phones' to browse our catalog!"
    )

    def __init__(self, faqs_path: str = "faqs.json", phones_path: str = "phones.json"):
        self.faqs = self._load_faqs(faqs_path)
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers   = [faq["answer"]   for faq in self.faqs]

        # Phone catalog
        self.catalog = PhoneCatalog(phones_path)

        # Pre-process all questions and build the TF-IDF matrix
        processed_questions = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)

        print(f"[Chatbot] Loaded {len(self.faqs)} FAQs and built TF-IDF index.")


    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _load_faqs(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                faqs = json.load(f)
            assert isinstance(faqs, list) and all("question" in x and "answer" in x for x in faqs)
            return faqs
        except (FileNotFoundError, AssertionError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to load FAQs from '{path}': {exc}") from exc

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def get_answer(self, user_input: str) -> dict:
        """
        Find the best-matching FAQ for *user_input*.

        Returns a dict:
          {
            "answer":     str,
            "matched_q":  str | None,   # the matched FAQ question
            "score":      float          # cosine similarity score (0-1)
          }
        """
        if not user_input.strip():
            return {"answer": "Please type a question!", "matched_q": None, "score": 0.0}

        # 1. Check for small talk / greetings first
        small_talk_reply = detect_small_talk(user_input)
        if small_talk_reply:
            return {"answer": small_talk_reply, "matched_q": None, "score": 1.0, "intent": "small_talk"}

        # 2. Check for phone catalog / price queries
        phone_reply = detect_phone_query(user_input, self.catalog)
        if phone_reply:
            return {"answer": phone_reply, "matched_q": None, "score": 1.0, "intent": "phone_catalog"}

        # 3. TF-IDF cosine similarity FAQ matching
        processed_input = preprocess(user_input)
        if not processed_input.strip():
            # All tokens were stop words / punctuation
            return {"answer": self.FALLBACK_RESPONSE, "matched_q": None, "score": 0.0}

        query_vec     = self.vectorizer.transform([processed_input])
        similarities  = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx      = int(similarities.argmax())
        best_score    = float(similarities[best_idx])

        if best_score >= self.SIMILARITY_THRESHOLD:
            return {
                "answer":    self.answers[best_idx],
                "matched_q": self.questions[best_idx],
                "score":     round(best_score, 4),
            }
        return {"answer": self.FALLBACK_RESPONSE, "matched_q": None, "score": round(best_score, 4)}




# --------------------------------------------------------------------------- #
# Quick CLI test (run: python chatbot.py)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    # Reconfigure stdout to UTF-8 so emoji don't crash Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bot = FAQChatbot()
    print("\n=== FAQ Chatbot CLI Test ===")
    print("Type 'quit' to exit.\n")

    test_queries = [
        # --- Small talk / greetings ---
        "hi",
        "Hello there!",
        "thanks a lot",
        "bye bye",
        "Who are you?",
        "what can you do?",
        "okay",
        # --- FAQ matching ---
        "My battery dies really quickly",
        "How can I take good pictures?",
        "What is 5G?",
        "My phone is getting really hot",
        "How do I reset my device?",
        "Tell me about OLED vs LCD",
        "How do I take a screenshot?",
        "How do I find my lost phone?",
        "What's the weather today?",   # off-topic → fallback
    ]

    for query in test_queries:
        result = bot.get_answer(query)
        intent = result.get("intent", "faq")
        print(f"Q: {query}")
        if intent == "small_talk":
            print(f"   [SMALL TALK detected]")
        else:
            print(f"   Matched: {result['matched_q']}  (score={result['score']})")
        print(f"   A: {result['answer']}\n")

