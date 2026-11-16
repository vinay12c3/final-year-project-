"""Turns a sequence of recognised signs into a natural English sentence.

ISL is not signed word-for-word like English: a signer shows the key signs
("hello", "how are you") and the listener fills in the grammar. This module
does that filling-in with a small, transparent set of rules, so the output is
predictable and runs instantly on any laptop (no language model needed).

A sign sequence is a list of tokens:
    {"kind": "sentence", "label": "s_hi_how_are_you", "text": "Hi, how are you?"}   a whole sentence sign
    {"kind": "word", "label": "how_are_you"}   a word sign from the words model
    {"kind": "spell", "text": "RAVI"}          a fingerspelled word (letters mode)

Examples
    hello, how_are_you          -> "Hello, how are you?"
    good_morning, name          -> "Good morning, what is your name?"
    name, RAVI                  -> "My name is Ravi."
    I, AM, FINE                 -> "I am fine."
    please, water               -> "Please, I need water."
    thank_you, pleased          -> "Thank you. Pleased to meet you."
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as C

GREETING, QUESTION, STATEMENT, POLITE = "greeting", "question", "statement", "polite"
QUESTION_WORDS = {"what", "where", "when", "who", "why", "how", "which", "can", "do", "is", "are"}


@dataclass(frozen=True)
class Phrase:
    text: str
    kind: str


# What each word sign means in a conversation.
PHRASES: dict[str, Phrase] = {
    "hello": Phrase("hello", GREETING),
    "good_morning": Phrase("good morning", GREETING),
    "good_afternoon": Phrase("good afternoon", GREETING),
    "good_evening": Phrase("good evening", GREETING),
    "good_night": Phrase("good night", GREETING),
    "how_are_you": Phrase("how are you", QUESTION),
    "name": Phrase("what is your name", QUESTION),
    "alright": Phrase("I am alright", STATEMENT),
    "thank_you": Phrase("thank you", STATEMENT),
    "pleased": Phrase("pleased to meet you", STATEMENT),
    "sorry": Phrase("sorry", STATEMENT),
    "yes": Phrase("yes", STATEMENT),
    "no": Phrase("no", STATEMENT),
    "help": Phrase("I need help", STATEMENT),
    "doctor": Phrase("I need a doctor", STATEMENT),
    "water": Phrase("I need water", STATEMENT),
    "food": Phrase("I need food", STATEMENT),
    "please": Phrase("please", POLITE),
}


def _phrase_for(token: dict) -> Phrase | None:
    if token.get("kind") == "sentence":
        # a whole signed sentence: used as it is
        text = str(token.get("text", "")).strip()
        if not text:
            return None
        return Phrase(text.rstrip(".!?"), QUESTION if text.endswith("?") else STATEMENT)
    if token.get("kind") == "spell":
        text = str(token.get("text", "")).strip()
        return Phrase(text.capitalize(), STATEMENT) if text else None
    label = str(token.get("label", ""))
    if not label or C.is_background(label):
        return None
    return PHRASES.get(label, Phrase(C.display_label(label), STATEMENT))


def _spelled_run(tokens: list[dict], i: int) -> tuple[list[str], int]:
    """Collects consecutive fingerspelled words starting at i: (words, next index)."""
    words = []
    while i < len(tokens) and tokens[i].get("kind") == "spell":
        w = str(tokens[i].get("text", "")).strip()
        if w:
            words.append(w)
        i += 1
    return words, i


def _clauses(tokens: list[dict]) -> list[Phrase]:
    """Applies the combination rules and returns the clauses in order."""
    out: list[Phrase] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        # "name" followed by spelled words introduces the signer: My name is Ravi Kumar.
        if tok.get("kind") == "word" and tok.get("label") == "name" and i + 1 < len(tokens) \
                and tokens[i + 1].get("kind") == "spell":
            words, i = _spelled_run(tokens, i + 1)
            if words:
                out.append(Phrase("my name is " + " ".join(w.capitalize() for w in words), STATEMENT))
            continue
        # spelled words in a row form one clause: I, AM, FINE -> "I am fine."
        if tok.get("kind") == "spell":
            words, i = _spelled_run(tokens, i)
            if words:
                text = " ".join(w.lower() for w in words)
                text = " ".join("I" if w == "i" else w for w in text.split())
                kind = QUESTION if words[0].lower() in QUESTION_WORDS else STATEMENT
                out.append(Phrase(text, kind))
            continue
        p = _phrase_for(tok)
        if p is not None and not (out and out[-1] == p):     # drop an accidental repeat
            out.append(p)
        i += 1
    return out


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _decap(s: str) -> str:
    """Lower-cases the first letter after a comma, except for the word "I"."""
    return s if not s or s == "I" or s.startswith(("I ", "I'")) else s[:1].lower() + s[1:]


def compose(tokens: list[dict]) -> str:
    """Builds the sentence for a sign sequence ('' when there is nothing to say)."""
    clauses = _clauses(tokens)
    parts: list[str] = []
    start = True          # next clause starts a new sentence
    pending_polite = False
    for k, p in enumerate(clauses):
        last = k == len(clauses) - 1
        if p.kind == POLITE:
            if last and parts and parts[-1][-1] in ".?!":
                # a closing "please" softens the sentence before it: "I need help, please."
                parts[-1] = parts[-1][:-1] + ", please" + parts[-1][-1]
            elif last:
                parts.append((" " if parts else "") + ("Please." if start else "please."))
            else:
                pending_polite = True
            continue
        text = p.text
        if pending_polite:
            text = "please, " + text
            pending_polite = False
        text = _cap(text) if start else _decap(text)
        sep = " " if parts else ""
        if p.kind == GREETING and not last:
            # greetings run into what follows: "Hello, how are you?"
            parts.append(sep + text + ",")
            start = False
            continue
        end = "?" if p.kind == QUESTION else "!" if p.kind == GREETING else "."
        parts.append(sep + text + end)
        start = True
    sentence = "".join(parts).strip()
    return sentence[:-1] + "." if sentence.endswith(",") else sentence
