"""Safety checks for a children's tutor (design §7).

Two checks per turn:
- input (the student's message, before any model sees it): personal data is scrubbed; prompt-injection attempts are
  refused; harmful or risk-to-life messages get a fixed reply and are flagged.
- output (the tutor's reply, while it streams): a reply that states a protected quiz answer or contains harmful
  phrases is stopped before that text reaches the student and replaced with a fixed reply. After the reply, a
  grounding score (how much of it comes from the lesson passages) is computed and a low score is flagged.

The words, patterns and fixed replies are content in ai/safety/vN.toml (DRAFT until the safety owner signs off), not
code. This layer is deterministic and explainable and works with every provider; a model-based classifier is added
for production (design §7).

Nothing here logs the student's text.
"""

import hashlib
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Literal

Severity = Literal["none", "low", "medium", "high", "critical"]
SEVERITY_RANK: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
REQUIRED_REPLIES = frozenset(
    {"off_topic", "personal_data", "injection", "high", "critical", "output_fallback", "quiz_fallback"}
)
REPLY_PLACEHOLDERS: dict[str, frozenset[str]] = {"off_topic": frozenset({"lesson_title"})}

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what when where which "
    "who why will with you your do does did can so if not no yes".split()
)

# Personal data (design §7). Indian mobile numbers (optionally +91 / 0 prefixed, spaces or dashes inside), other long
# digit runs that look like phone numbers, e-mail addresses, and "my address is …" / "I live at …" statements.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("phone", re.compile(r"(?<![\w+])(?:\+?91[\s-]?|0)?[6-9](?:[\s-]?\d){9}(?!\d)")),
    ("phone", re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?\d(?:[\s-]?\d){5,11}(?!\d)")),
    ("address", re.compile(r"\b(?:my address is|i live at|my house is at)\b[^.!?\n]*", re.IGNORECASE)),
)


class SafetyPolicyError(RuntimeError):
    pass


def normalize(text: str) -> str:
    """Lower-case, punctuation to single spaces: 'Kill MYSELF!!' -> 'kill myself'."""
    return _NON_WORD.sub(" ", text.casefold()).strip()


def content_words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w not in _STOPWORDS and len(w) > 1]


def contains_phrase(normalized_text: str, normalized_phrase: str) -> bool:
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


def scrub_pii(text: str) -> tuple[str, tuple[str, ...]]:
    """Replace personal data with placeholders; returns the new text and the kinds found (e.g. ("phone",))."""
    kinds: list[str] = []
    for kind, pattern in _PII_PATTERNS:
        text, count = pattern.subn(f"[{kind} hidden]", text)
        if count and kind not in kinds:
            kinds.append(kind)
    return text, tuple(kinds)


def leaked_answer(text: str, protected: Iterable[str]) -> str | None:
    """The first protected answer the text states (whole words, ignoring case and punctuation), if any."""
    normalized = normalize(text)
    for answer in protected:
        if contains_phrase(normalized, normalize(answer)):
            return answer
    return None


def grounding_score(reply: str, sources: Sequence[str]) -> float:
    """Share of the reply's content words that appear in the sources (lesson passages and the student's message).
    1.0 for a reply with no content words."""
    words = content_words(reply)
    if not words:
        return 1.0
    known = {w for source in sources for w in content_words(source)}
    return sum(1 for w in words if w in known) / len(words)


@dataclass(frozen=True)
class Verdict:
    category: str = "ok"
    severity: Severity = "none"

    @property
    def flagged(self) -> bool:
        return SEVERITY_RANK[self.severity] >= SEVERITY_RANK["medium"]

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "severity": self.severity, "flagged": self.flagged}


OK = Verdict()


@dataclass(frozen=True)
class InputDecision:
    action: Literal["allow", "block"]
    text: str  # what the model may see (personal data scrubbed)
    verdict: Verdict
    reply: str | None = None  # fixed reply when blocked
    notice: str | None = None  # shown before the tutor's answer when allowed (e.g. "don't share personal data")
    personal_data: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Category:
    name: str
    severity: Severity
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class SafetyPolicy:
    version: str
    checksum: str
    replies: dict[str, Template]
    categories: tuple[_Category, ...]
    injection: tuple[re.Pattern[str], ...]
    injection_severity: Severity
    longest_phrase: int = field(default=0)

    @property
    def ref(self) -> str:
        return f"safety/{self.version}"

    @classmethod
    def load(cls, path: Path) -> "SafetyPolicy":
        try:
            raw = path.read_bytes()
            data = tomllib.loads(raw.decode("utf-8"))
        except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            raise SafetyPolicyError(f"cannot read safety policy {path}: {exc}") from exc
        return cls.from_dict(data, checksum=hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_dict(cls, data: dict[str, Any], checksum: str = "") -> "SafetyPolicy":
        version = str(data.get("version") or "")
        if not re.fullmatch(r"v[1-9][0-9]*", version):
            raise SafetyPolicyError('safety policy needs version = "vN"')
        replies_raw = data.get("replies") or {}
        missing = REQUIRED_REPLIES - set(replies_raw)
        if missing:
            raise SafetyPolicyError(f"safety policy is missing replies {sorted(missing)}")
        replies: dict[str, Template] = {}
        for name, text in replies_raw.items():
            template = Template(str(text))
            unknown = set(template.get_identifiers()) - REPLY_PLACEHOLDERS.get(name, frozenset())
            if not template.is_valid() or unknown or not str(text).strip():
                raise SafetyPolicyError(f"safety reply {name!r} is empty or has unknown placeholders")
            replies[name] = template
        categories = []
        for name, spec in (data.get("categories") or {}).items():
            severity = spec.get("severity")
            if severity not in ("low", "medium", "high", "critical"):
                raise SafetyPolicyError(f"category {name!r} has an unknown severity {severity!r}")
            phrases = tuple(p for p in (normalize(str(x)) for x in spec.get("phrases") or []) if p)
            if not phrases:
                raise SafetyPolicyError(f"category {name!r} has no phrases")
            categories.append(_Category(name, severity, phrases))
        injection_spec = data.get("injection") or {}
        try:
            patterns = tuple(re.compile(p, re.IGNORECASE) for p in injection_spec.get("patterns") or [])
        except re.error as exc:
            raise SafetyPolicyError(f"bad injection pattern: {exc}") from exc
        injection_severity = injection_spec.get("severity", "medium")
        if injection_severity not in SEVERITY_RANK:
            raise SafetyPolicyError(f"injection has an unknown severity {injection_severity!r}")
        longest = max((len(p) for c in categories for p in c.phrases), default=0)
        return cls(version, checksum, replies, tuple(categories), patterns, injection_severity, longest)

    # -- replies ------------------------------------------------------------------------------------------------

    def reply(self, name: str, **values: str) -> str:
        return self.replies[name].substitute(values)

    # -- classification -----------------------------------------------------------------------------------------

    def classify(self, text: str) -> Verdict:
        """The most severe category whose phrases the text contains (OK if none)."""
        normalized = normalize(text)
        best = OK
        for category in self.categories:
            if SEVERITY_RANK[category.severity] <= SEVERITY_RANK[best.severity]:
                continue
            if any(contains_phrase(normalized, phrase) for phrase in category.phrases):
                best = Verdict(category.name, category.severity)
        return best

    def is_injection(self, text: str) -> bool:
        lowered = " ".join(text.casefold().split())
        return any(pattern.search(lowered) for pattern in self.injection)

    def check_input(self, message: str) -> InputDecision:
        scrubbed, personal = scrub_pii(message)
        verdict = self.classify(message)
        if verdict.severity == "critical":
            return InputDecision("block", scrubbed, verdict, reply=self.reply("critical"), personal_data=personal)
        if verdict.severity == "high":
            return InputDecision("block", scrubbed, verdict, reply=self.reply("high"), personal_data=personal)
        if self.is_injection(message):
            injection = Verdict("injection", self.injection_severity)
            return InputDecision("block", scrubbed, injection, reply=self.reply("injection"), personal_data=personal)
        if personal:
            return InputDecision(
                "allow",
                scrubbed,
                Verdict("personal_data", "medium"),
                notice=self.reply("personal_data"),
                personal_data=personal,
            )
        return InputDecision("allow", scrubbed, verdict)


class StreamGuard:
    """Checks a streaming reply and releases text with a lag, so a protected answer or a harmful phrase is caught
    before any part of it reaches the student.

    The lag (`window` characters) is longer than any protected answer or policy phrase, allowing for extra spaces
    and punctuation between words. Everything not yet released is withheld once a problem is found.
    """

    def __init__(self, policy: SafetyPolicy, protected_answers: Sequence[str], margin: int = 20) -> None:
        self.policy = policy
        self.protected = [a for a in protected_answers if normalize(a)]
        longest = max([policy.longest_phrase, *(len(a) for a in self.protected)], default=0)
        self.window = 2 * longest + margin
        self.text = ""
        self.released = 0
        self.problem: Literal["quiz_answer", "harmful"] | None = None
        self.verdict: Verdict = OK

    def feed(self, delta: str) -> str:
        """Add model output; returns the text that is now safe to release ('' when withheld or stopped)."""
        if self.problem is not None:
            return ""
        self.text += delta
        # Optimization: Evaluate trailing text window to avoid O(N^2) rescan while preserving cross-chunk detection
        scan_window = self.text[max(0, len(self.text) - max(2 * self.window, len(delta) + self.window)) :]
        if leaked_answer(scan_window, self.protected) is not None:
            self.problem, self.verdict = "quiz_answer", Verdict("quiz_answer", "medium")
            return ""
        verdict = self.policy.classify(scan_window)
        if SEVERITY_RANK[verdict.severity] >= SEVERITY_RANK["high"]:
            self.problem, self.verdict = "harmful", verdict
            return ""
        return self._release(len(self.text) - self.window)

    def finish(self) -> str:
        """The rest of the text, once the reply is complete and passed every check."""
        return "" if self.problem is not None else self._release(len(self.text))

    def _release(self, upto: int) -> str:
        if upto <= self.released:
            return ""
        chunk = self.text[self.released : upto]
        self.released = upto
        return chunk
