"""
Week 5 – Task 4: Personalized Fashion Recommendations (RAG Pipeline)
======================================================================
Retrieval-Augmented Generation for fashion advice:
  1. User provides body type, occasion, mood, budget, preferences
  2. Retriever fetches relevant styles/trends/elements from ChromaDB
  3. Generator synthesizes a personalized recommendation

This is the full RAG loop:
  User Profile → Query Construction → ChromaDB Retrieval → LLM Generation
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from task3_chromadb_retrieval import FashionRetriever


# ─────────────────────────────────────────────
# USER PROFILE
# ─────────────────────────────────────────────

@dataclass
class UserProfile:
    name: str
    body_type: str                          # e.g. "pear", "hourglass", "petite", "tall"
    occasion: str                           # e.g. "office", "wedding guest", "casual weekend"
    mood: str                               # e.g. "confident", "romantic", "playful"
    preferred_colors: List[str] = field(default_factory=list)
    disliked_colors: List[str] = field(default_factory=list)
    budget: str = "mid-range"              # "budget", "mid-range", "luxury"
    style_history: List[str] = field(default_factory=list)   # past styles they liked
    season: str = "all-season"

    def to_query(self) -> str:
        """Convert profile into a semantic search query string."""
        parts = [
            f"{self.mood} {self.occasion} look",
            f"{self.body_type} body type",
            f"{self.season} fashion",
        ]
        if self.preferred_colors:
            parts.append(f"colors: {', '.join(self.preferred_colors)}")
        if self.style_history:
            parts.append(f"style preferences: {', '.join(self.style_history)}")
        return ". ".join(parts)


# ─────────────────────────────────────────────
# RAG RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

class FashionRecommendationEngine:
    """
    Full RAG pipeline for personalized fashion recommendations.

    Retrieval: ChromaDB semantic search
    Generation: Rule-based (production) or LLM (if API key available)
    """

    def __init__(
        self,
        persist_dir: str = "database/chromadb",
        embeddings_path: str = "embeddings/fashion_embeddings.json",
        collection_name: str = "fashion_styles",
    ):
        self.retriever = FashionRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embeddings_path=embeddings_path,
        )

    # ── RETRIEVAL ────────────────────────────────────────────────────────

    def retrieve_context(
        self, profile: UserProfile, top_k: int = 3
    ) -> Dict[str, List[Dict]]:
        """
        Multi-query retrieval: separate queries for styles, trends, elements.
        Returns a dict with lists under each type key.
        """
        base_query = profile.to_query()

        styles   = self.retriever.query(base_query, n_results=top_k, filter_type="style")
        trends   = self.retriever.query(base_query, n_results=2,     filter_type="trend")
        elements = self.retriever.query(base_query, n_results=2,     filter_type="design_element")
        occasions = self.retriever.query(profile.occasion, n_results=1, filter_type="occasion")

        return {
            "styles"  : styles,
            "trends"  : trends,
            "elements": elements,
            "occasion": occasions,
        }

    # ── GENERATION ───────────────────────────────────────────────────────

    def _build_prompt(
        self, profile: UserProfile, context: Dict[str, List[Dict]]
    ) -> str:
        """Builds the LLM prompt from retrieved context + user profile."""

        ctx_lines = []
        for category, items in context.items():
            if items:
                ctx_lines.append(f"\n## Retrieved {category.title()}:")
                for item in items:
                    ctx_lines.append(f"  - {item['name']} (score: {item['similarity']:.2f}): {item.get('description','')[:120]}")

        context_block = "\n".join(ctx_lines)

        prompt = f"""You are an expert AI Fashion Stylist with deep knowledge of trends, body types, and personal styling.

## User Profile:
- Name           : {profile.name}
- Body Type      : {profile.body_type}
- Occasion       : {profile.occasion}
- Mood           : {profile.mood}
- Budget         : {profile.budget}
- Season         : {profile.season}
- Preferred Colors: {', '.join(profile.preferred_colors) or 'No preference'}
- Disliked Colors : {', '.join(profile.disliked_colors) or 'None'}
- Past Style Likes: {', '.join(profile.style_history) or 'Not specified'}

## Style Intelligence Retrieved from Knowledge Base:
{context_block}

## Your Task:
Using the retrieved style intelligence above, provide a highly personalized fashion recommendation for {profile.name}. Structure your response as:

1. **Your Style Identity** – A 2-sentence style persona for this person
2. **Recommended Outfit** – Specific pieces, colors, silhouettes
3. **Why This Works for You** – Body type + occasion justification
4. **Trending Element to Incorporate** – One current trend that fits their profile
5. **Styling Tips** – 3 actionable tips
6. **Brands to Explore** – 3 brands at their budget level

Be warm, specific, and encouraging. Reference the retrieved styles by name where relevant."""

        return prompt

    def generate_rule_based(
        self, profile: UserProfile, context: Dict[str, List[Dict]]
    ) -> str:
        """
        Fallback rule-based recommendation (no LLM needed).
        Production systems would replace this with an LLM call.
        """
        top_style   = context["styles"][0]   if context["styles"]   else {"name": "Classic"}
        top_trend   = context["trends"][0]   if context["trends"]   else {"name": "Timeless"}
        top_element = context["elements"][0] if context["elements"] else {"name": "Clean silhouette"}
        occasion_tip = context["occasion"][0] if context["occasion"] else {}

        colors_str = ", ".join(profile.preferred_colors) if profile.preferred_colors else "neutrals and accent tones"

        # Body type → silhouette map
        silhouettes = {
            "pear"      : "A-line skirts and wide-leg trousers that balance your proportions",
            "hourglass" : "Wrap dresses and belted styles that celebrate your waist",
            "petite"    : "Vertical lines, monochromatic looks, and cropped proportions to elongate",
            "tall"      : "Maxi lengths, wide belts, and horizontal details that work with your height",
            "athletic"  : "Ruched details, peplum tops, and draped fabrics to add curves",
            "plus"      : "Empire waists, V-necks, and structured pieces that flatter your figure",
        }
        silhouette = silhouettes.get(
            profile.body_type.lower(),
            "pieces that highlight your best features"
        )

        # Budget → brand tier
        brand_tiers = {
            "budget"   : ["Zara", "H&M", "ASOS", "Mango"],
            "mid-range": ["& Other Stories", "Reiss", "COS", "Sandro"],
            "luxury"   : ["Toteme", "Max Mara", "Zimmermann", "Reformation"],
        }
        brands = brand_tiers.get(profile.budget, brand_tiers["mid-range"])

        rec = f"""
╔══════════════════════════════════════════════════════════════╗
  ✨  Personalized Fashion Recommendation for {profile.name}
╚══════════════════════════════════════════════════════════════╝

1. YOUR STYLE IDENTITY
   Your aesthetic leans towards {top_style['name']}, bringing a sense of 
   {profile.mood} energy to every look. You dress with intention — every 
   piece chosen for both confidence and occasion.

2. RECOMMENDED OUTFIT FOR: {profile.occasion.upper()}
   • Silhouette  : {silhouette}
   • Color Palette: {colors_str}
   • Key Pieces  :
       - A beautifully tailored {top_style['name'].lower()} top or blouse
       - Complementary bottoms in your palette ({colors_str.split(',')[0].strip()})
       - Shoes and a bag that echo the {profile.mood} mood
       - One statement accessory that elevates the whole look

3. WHY THIS WORKS FOR YOU
   For a {profile.body_type} body type, {silhouette}. 
   The {top_style['name']} aesthetic aligns perfectly with the {profile.occasion} 
   setting, striking the right balance of polish and personality.

4. TRENDING ELEMENT TO INCORPORATE
   Right now, {top_trend['name']} is everywhere. You can add this trend to your 
   look with one subtle piece — keeping your overall aesthetic grounded while 
   feeling current.

5. STYLING TIPS
   ① Anchor your outfit with {colors_str.split(',')[0].strip() if profile.preferred_colors else 'a neutral base'} 
     as the dominant color and use accents sparingly.
   ② Incorporate a {top_element['name'].lower()} detail — it adds visual interest 
     without overwhelming the look.
   ③ {occasion_tip.get('description', 'Layer thoughtfully for versatility throughout the day.')[:120] if occasion_tip else 'Focus on fit above all else — well-fitted basics always outperform trend pieces.'}

6. BRANDS TO EXPLORE ({profile.budget.upper()} BUDGET)
   {' · '.join(brands)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Retrieved context: {top_style['name']} style  ·  {top_trend['name']} trend
  Similarity scores: {top_style['similarity']:.2f}  ·  {top_trend['similarity']:.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return rec

    # ── MAIN RECOMMEND METHOD ─────────────────────────────────────────────

    def recommend(
        self,
        profile: UserProfile,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None
    ) -> str:
        """
        Full RAG recommendation pipeline.

        Args:
            profile     : UserProfile instance
            use_llm     : If True, calls an LLM with the retrieved context
            llm_api_key : API key (OpenAI or Anthropic) if use_llm=True
        """
        print(f"\n🔄 Retrieving style intelligence for {profile.name}...")
        context = self.retrieve_context(profile)

        if use_llm and llm_api_key:
            try:
                import anthropic
                prompt = self._build_prompt(profile, context)
                client = anthropic.Anthropic(api_key=llm_api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                print(f"⚠️  LLM call failed ({e}), falling back to rule-based.")

        return self.generate_rule_based(profile, context)


# ─────────────────────────────────────────────
# DEMO – 3 different user profiles
# ─────────────────────────────────────────────

DEMO_PROFILES = [
    UserProfile(
        name="Priya",
        body_type="hourglass",
        occasion="wedding guest",
        mood="romantic",
        preferred_colors=["blush", "champagne", "ivory"],
        disliked_colors=["black", "grey"],
        budget="mid-range",
        style_history=["bohemian", "minimalist"],
        season="summer"
    ),
    UserProfile(
        name="Arjun",
        body_type="athletic",
        occasion="office",
        mood="confident",
        preferred_colors=["navy", "white", "olive"],
        budget="mid-range",
        style_history=["streetwear", "minimalist"],
        season="all-season"
    ),
    UserProfile(
        name="Neha",
        body_type="petite",
        occasion="casual weekend",
        mood="playful",
        preferred_colors=["hot pink", "lavender", "white"],
        budget="budget",
        style_history=["Y2K Revival", "streetwear"],
        season="spring"
    ),
]


def run_demo():
    print("=" * 65)
    print("  Week 5 · Task 4 — Personalized Fashion Recommendations (RAG)")
    print("=" * 65)

    engine = FashionRecommendationEngine()

    for profile in DEMO_PROFILES:
        print(f"\n{'━'*65}")
        recommendation = engine.recommend(profile, use_llm=False)
        print(recommendation)

    print("\n✅ Demo complete! To use LLM generation:")
    print("   engine.recommend(profile, use_llm=True, llm_api_key='your-key')")


if __name__ == "__main__":
    run_demo()
