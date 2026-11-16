"""Sentence signs from the ISL-CSLTR corpus (Elakkiya & Natarajan, 2021, CC BY 4.0).

Each sentence is signed as one continuous movement, so the app treats a whole
sentence as one class: the signer shows the sentence and the app writes it.
The corpus folder names are turned into a stable label and readable text here.
"""
from __future__ import annotations

import re

QUESTION_START = ("are ", "can ", "could ", "do ", "had ", "how ", "shall ", "what ", "when ", "where ",
                  "which ", "who ", "why ", "is ")

# Folder names in the corpus that need a little cleaning to read well.
FIXES = {
    "no need to worry dont worry": "No need to worry, don't worry",
    "he she is my friend": "He/she is my friend",
    "i am (age)": "I am ___ years old",
    "my name is xxxxxxxx": "My name is ___",
    "i dont agree": "I don't agree",
    "which collegeschool are you from": "Which college/school are you from?",
    "i am fine. thank you sir": "I am fine. Thank you, sir.",
    "i like you i love you": "I like you, I love you",
    "turn on light turn off light": "Turn on the light / turn off the light",
    "had your food": "Had your food?",
    "you do anything, i do not care": "You do anything, I do not care",
    "you need a medicine, take this one": "You need a medicine, take this one",
    "hi how are you": "Hi, how are you?",
}


def label_for(folder: str) -> str:
    """'Hi how are you' -> 's_hi_how_are_you' (prefix keeps sentences apart from words)."""
    slug = re.sub(r"[^a-z0-9]+", "_", folder.strip().lower()).strip("_")
    return "s_" + slug


def is_sentence(label: str) -> bool:
    return label.startswith("s_")


def text_for(folder: str) -> str:
    key = re.sub(r"\s+", " ", folder.strip().lower())
    if key in FIXES:
        text = FIXES[key]
    else:
        text = re.sub(r"\bi\b", "I", key)
        text = text[:1].upper() + text[1:]
    if text[-1] not in ".?!_":
        question = key.startswith(QUESTION_START) and not key.startswith(("do not", "do me"))
        text += "?" if question else "."
    return text
