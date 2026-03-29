"""
LucilleLLM - Cultural Competence Service

Provides bias detection in LLM outputs, cultural context awareness
for prompt engineering, and diverse international crisis resources.
All checks use keyword/pattern matching -- no additional LLM calls.

No Firestore writes -- reads user profile's cultural_background.

Follows the singleton pattern from other services.
"""

import logging
import re
from typing import List, Optional, Tuple

from config import get_config
from models import CrisisResource, CrisisResourceType, CulturalContext

logger = logging.getLogger(__name__)


# ── International Crisis Resources ───────────────────────
# Organized by country code (ISO 3166-1 alpha-2)

INTERNATIONAL_CRISIS_RESOURCES = {
    "US": [
        CrisisResource(
            name="988 Suicide & Crisis Lifeline",
            number="988",
            description="Free, confidential 24/7 support.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://988lifeline.org",
            country="US",
        ),
        CrisisResource(
            name="Crisis Text Line",
            number="Text HOME to 741741",
            description="Free 24/7 crisis counseling via text.",
            resource_type=CrisisResourceType.TEXT,
            url="https://www.crisistextline.org",
            country="US",
        ),
    ],
    "GB": [
        CrisisResource(
            name="Samaritans",
            number="116 123",
            description="Free 24/7 emotional support.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.samaritans.org",
            country="GB",
        ),
        CrisisResource(
            name="Shout",
            number="Text SHOUT to 85258",
            description="Free 24/7 text support.",
            resource_type=CrisisResourceType.TEXT,
            url="https://giveusashout.org",
            country="GB",
        ),
    ],
    "CA": [
        CrisisResource(
            name="Talk Suicide Canada",
            number="988",
            description="24/7 suicide prevention.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://talksuicide.ca",
            country="CA",
        ),
        CrisisResource(
            name="Crisis Text Line Canada",
            number="Text HOME to 686868",
            description="Free 24/7 text support.",
            resource_type=CrisisResourceType.TEXT,
            url="https://www.crisistextline.ca",
            country="CA",
        ),
    ],
    "AU": [
        CrisisResource(
            name="Lifeline Australia",
            number="13 11 14",
            description="24/7 crisis support.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.lifeline.org.au",
            country="AU",
        ),
        CrisisResource(
            name="Beyond Blue",
            number="1300 22 4636",
            description="Mental health support.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.beyondblue.org.au",
            country="AU",
        ),
    ],
    "IN": [
        CrisisResource(
            name="iCall",
            number="9152987821",
            description="Professional counseling service.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://icallhelpline.org",
            country="IN",
        ),
        CrisisResource(
            name="Vandrevala Foundation",
            number="1860-2662-345",
            description="24/7 mental health helpline.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.vandrevalafoundation.com",
            country="IN",
        ),
    ],
    "DE": [
        CrisisResource(
            name="Telefonseelsorge",
            number="0800 111 0 111",
            description="24/7 crisis counseling (free).",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.telefonseelsorge.de",
            country="DE",
        ),
    ],
    "JP": [
        CrisisResource(
            name="TELL Lifeline",
            number="03-5774-0992",
            description="English-language counseling in Japan.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://telljp.com",
            country="JP",
        ),
    ],
    "BR": [
        CrisisResource(
            name="CVV (Centro de Valorização da Vida)",
            number="188",
            description="24/7 emotional support.",
            resource_type=CrisisResourceType.HOTLINE,
            url="https://www.cvv.org.br",
            country="BR",
        ),
    ],
    "INTL": [
        CrisisResource(
            name="Befrienders Worldwide",
            number="",
            description="International crisis center directory.",
            resource_type=CrisisResourceType.WEBSITE,
            url="https://www.befrienders.org/find-a-helpline",
            country="INTL",
        ),
        CrisisResource(
            name="International Association for Suicide Prevention",
            number="",
            description="Global crisis center directory.",
            resource_type=CrisisResourceType.WEBSITE,
            url="https://www.iasp.info/resources/Crisis_Centres/",
            country="INTL",
        ),
    ],
}


# ── Bias Detection Patterns ──────────────────────────────
# Patterns indicating cultural insensitivity or stereotyping in LLM output

STEREOTYPE_PATTERNS = [
    r"(people\s+from|in)\s+(africa|asia|india|china|mexico|middle\s+east)\s+(are|tend\s+to|always|usually)\b",
    r"\b(all|every)\s+(asians?|africans?|hispanics?|latinos?|indians?|muslims?|jews?|christians?)\s+(are|do|have)\b",
    r"\byour\s+(culture|people|country)\s+(doesn't|don't|isn't|aren't)\s+value\b",
    r"\bthat's\s+(just|typical)\s+(your|their)\s+culture\b",
    r"\bback\s+in\s+your\s+country\b",
    r"\byou\s+people\b",
]

CULTURAL_INSENSITIVITY_PATTERNS = [
    r"(just\s+)?pray\s+(about\s+it|more|harder)",
    r"(meditation|yoga|mindfulness)\s+will\s+(fix|cure|solve)\s+everything",
    r"your\s+family\s+(should|needs\s+to|must)\s+(understand|accept|support)",
    r"(stop|quit)\s+being\s+so\s+(dramatic|emotional|sensitive)\s+about\s+(your|the)\s+(religion|culture|tradition)",
    r"just\s+leave\s+(your|the)\s+(family|community|religion|culture)",
    r"that('s|\s+is)\s+not\s+how\s+(we|things)\s+work\s+here",
]

WESTERN_CENTRIC_ASSUMPTIONS = [
    r"(everyone|you)\s+should\s+see\s+a\s+therapist",
    r"talk\s+to\s+your\s+(therapist|psychiatrist|counselor)",
    r"individual\s+therapy\s+is\s+the\s+(best|only)\s+way",
    r"you\s+need\s+to\s+(set|establish)\s+boundaries\s+with\s+your\s+(parents|family|elders)",
]


# ── Cultural Background to Country Code Mapping ─────────

CULTURAL_COUNTRY_HINTS = {
    "american": "US", "united states": "US",
    "british": "GB", "uk": "GB", "english": "GB", "scottish": "GB", "welsh": "GB",
    "canadian": "CA", "canada": "CA",
    "australian": "AU", "australia": "AU",
    "indian": "IN", "india": "IN",
    "german": "DE", "germany": "DE",
    "japanese": "JP", "japan": "JP",
    "brazilian": "BR", "brazil": "BR",
    "mexican": "US", "mexico": "US",  # Use US resources as closest regional match
    "chinese": "INTL", "china": "INTL",
    "korean": "INTL", "korea": "INTL",
    "french": "INTL", "france": "INTL",
    "spanish": "INTL", "spain": "INTL",
    "italian": "INTL", "italy": "INTL",
    "nigerian": "INTL", "nigeria": "INTL",
    "south african": "INTL", "south africa": "INTL",
    "pakistani": "IN", "pakistan": "IN",  # Nearest regional match
    "bangladeshi": "IN", "bangladesh": "IN",
    "sri lankan": "IN", "sri lanka": "IN",
    "filipino": "INTL", "philippines": "INTL",
}


# ── Service ──────────────────────────────────────────────

class CulturalService:
    """
    Service for cultural competence checks, bias detection,
    and international crisis resource management.

    No Firestore writes -- reads user profile's cultural_background.
    """

    def __init__(self):
        self._config = get_config()

    # ── Bias Detection (output validation) ───────────────

    def check_output_for_bias(self, bot_response: str) -> Tuple[List[str], bool]:
        """
        Scan bot response for cultural bias patterns.
        Returns (list_of_flags, is_culturally_appropriate).
        Fast: pure regex, no API calls.
        """
        if not self._config.CULTURAL_BIAS_CHECK_ENABLED:
            return [], True

        flags: List[str] = []
        response_lower = bot_response.lower()

        for pattern in STEREOTYPE_PATTERNS:
            if re.search(pattern, response_lower):
                flags.append(f"stereotype_pattern:{pattern[:40]}")

        for pattern in CULTURAL_INSENSITIVITY_PATTERNS:
            if re.search(pattern, response_lower):
                flags.append(f"cultural_insensitivity:{pattern[:40]}")

        for pattern in WESTERN_CENTRIC_ASSUMPTIONS:
            if re.search(pattern, response_lower):
                flags.append(f"western_centric:{pattern[:40]}")

        return flags, len(flags) == 0

    # ── Cultural Context Extraction ──────────────────────

    def extract_cultural_context(
        self, user_profile_data: Optional[dict]
    ) -> CulturalContext:
        """
        Extract cultural context from the user's profile.
        Used to select appropriate crisis resources and prompt context.
        """
        if not user_profile_data:
            return CulturalContext(
                country_code=self._config.CULTURAL_DEFAULT_COUNTRY,
            )

        user_id = user_profile_data.get("user_id", "")
        persona = user_profile_data.get("persona", {})
        cultural_bg = persona.get("cultural_background", "") if persona else ""

        country_code = self._config.CULTURAL_DEFAULT_COUNTRY
        if cultural_bg:
            bg_lower = cultural_bg.lower()
            for hint, code in CULTURAL_COUNTRY_HINTS.items():
                if hint in bg_lower:
                    country_code = code
                    break

        return CulturalContext(
            user_id=user_id,
            cultural_background=cultural_bg or "",
            country_code=country_code,
        )

    # ── International Crisis Resources ───────────────────

    def get_crisis_resources_for_country(
        self, country_code: str
    ) -> List[CrisisResource]:
        """
        Get crisis resources for a specific country.
        Falls back to US resources if country not found.
        Always includes INTL resources as supplement.
        """
        country_upper = country_code.upper() if country_code else "US"
        resources = INTERNATIONAL_CRISIS_RESOURCES.get(
            country_upper,
            INTERNATIONAL_CRISIS_RESOURCES["US"],  # default fallback
        )
        # Always append international resources (if not already INTL)
        if country_upper != "INTL":
            intl = INTERNATIONAL_CRISIS_RESOURCES.get("INTL", [])
            resources = resources + intl
        return resources

    def format_crisis_footer_for_country(self, country_code: str) -> str:
        """Format a culturally appropriate crisis resource footer."""
        resources = self.get_crisis_resources_for_country(country_code)
        lines = [
            "---",
            "**If you are in crisis or need immediate help, please reach out:**",
            "",
        ]
        for r in resources[:4]:  # Show up to 4 resources
            if r.number:
                lines.append(f"- **{r.name}** ({r.country}): {r.number}")
            else:
                lines.append(f"- **{r.name}**: [{r.url}]({r.url})")
        lines.append("")
        lines.append("*You are not alone. Help is available.*")
        return "\n".join(lines)

    # ── Cultural Context for Prompt Injection ────────────

    def get_cultural_prompt_context(
        self, cultural_context: CulturalContext
    ) -> str:
        """
        Generate cultural awareness text for injection into system prompt.
        Only added when a non-default cultural background is detected.
        """
        if not cultural_context.cultural_background:
            return ""

        return (
            f"CULTURAL AWARENESS: The user has indicated their cultural background "
            f"as '{cultural_context.cultural_background}'. Be mindful of:\n"
            "- Cultural differences in expressing emotions and seeking help\n"
            "- Family and community dynamics that may differ from Western norms\n"
            "- Religious or spiritual practices that may be important to them\n"
            "- Avoid imposing Western-centric therapeutic frameworks without context\n"
            "- Respect collectivist values if relevant (family obligations, community harmony)\n"
            "- Offer culturally appropriate coping strategies alongside evidence-based ones"
        )

    # ── Supported Countries List (for API/tool) ──────────

    def get_supported_countries(self) -> List[str]:
        """Return list of supported country codes."""
        return [k for k in INTERNATIONAL_CRISIS_RESOURCES.keys() if k != "INTL"]


# ── Singleton ────────────────────────────────────────────

_cultural_service: Optional[CulturalService] = None


def get_cultural_service() -> CulturalService:
    """Get or create CulturalService singleton."""
    global _cultural_service
    if _cultural_service is None:
        _cultural_service = CulturalService()
    return _cultural_service
