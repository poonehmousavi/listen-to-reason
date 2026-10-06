"""FLEURS, every language: a few development clips per language with their reference text.

    python -m l2r.dataset.fleurs --per-lang 40      ->  <data>/FLEURS/<code>/*.wav and <data>/FLEURS/meta.json

`meta.json` = {code: {"language": name, "group": geographic group, "clips": [{"path", "gender", "text"}]}}. The
group is the class of the tree's language attribute and the language its leaf. The file is read by the language
expert (language -> group), by the language head (reference sentences) and the pool takes its FLEURS clips from it.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from l2r.common import data

REPO = "google/fleurs"
GROUPS = {
    "western european": "Asturian Bosnian Catalan Croatian Danish Dutch English Finnish French Galician German Greek Hungarian Icelandic Irish "
                        "Italian Kabuverdianu Luxembourgish Maltese Norwegian Occitan Portuguese Spanish Swedish Welsh",
    "eastern european": "Armenian Belarusian Bulgarian Czech Estonian Georgian Latvian Lithuanian Macedonian Polish Romanian Russian Serbian "
                        "Slovak Slovenian Ukrainian",
    "central asian, middle eastern and north african": "Arabic Azerbaijani Hebrew Kazakh Kyrgyz Mongolian Pashto Persian Sorani-Kurdish Tajik Turkish Uzbek",
    "sub-saharan african": "Afrikaans Amharic Fula Ganda Hausa Igbo Kamba Lingala Luo Northern-Sotho Nyanja Oromo Shona Somali Swahili Umbundu "
                           "Wolof Xhosa Yoruba Zulu",
    "south asian": "Assamese Bengali Gujarati Hindi Kannada Malayalam Marathi Nepali Oriya Punjabi Sindhi Tamil Telugu Urdu",
    "south-east asian": "Burmese Cebuano Filipino Indonesian Javanese Khmer Lao Malay Maori Thai Vietnamese",
    "chinese, japanese and korean": "Cantonese Mandarin Japanese Korean"}
LANGUAGE = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "ast": "Asturian", "az": "Azerbaijani", "be": "Belarusian",
    "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian", "ca": "Catalan", "ceb": "Cebuano", "ckb": "Sorani-Kurdish", "cmn": "Mandarin",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
    "fa": "Persian", "ff": "Fula", "fi": "Finnish", "fil": "Filipino", "fr": "French", "ga": "Irish", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "ig": "Igbo",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian", "kam": "Kamba", "kea": "Kabuverdianu",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "ky": "Kyrgyz", "lb": "Luxembourgish", "lg": "Ganda", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "luo": "Luo", "lv": "Latvian", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "nb": "Norwegian", "ne": "Nepali", "nl": "Dutch", "nso": "Northern-Sotho",
    "ny": "Nyanja", "oc": "Occitan", "om": "Oromo", "or": "Oriya", "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sd": "Sindhi", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "umb": "Umbundu", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "wo": "Wolof", "xh": "Xhosa", "yo": "Yoruba", "yue": "Cantonese", "zu": "Zulu"}
GROUP_OF = {lang: g for g, langs in GROUPS.items() for lang in langs.split()}


def fetch(per_lang: int = 40) -> dict:
    """Download `per_lang` development clips of every language (one reading per sentence). Resumable per language."""
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files
    out = data("FLEURS"); out.mkdir(parents=True, exist_ok=True)
    mp = out / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    for code in sorted({f.split("/")[1] for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("data/")}):
        name = LANGUAGE.get(code.split("_")[0])
        if not name or name not in GROUP_OF or len(meta.get(code, {}).get("clips", [])) >= per_lang:
            continue
        tsv = hf_hub_download(REPO, f"data/{code}/dev.tsv", repo_type="dataset")
        tar = hf_hub_download(REPO, f"data/{code}/audio/dev.tar.gz", repo_type="dataset")
        rows = {}
        for line in open(tsv, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 7:
                rows[p[1]] = {"text": p[2], "gender": p[6].lower()}
        (out / code).mkdir(exist_ok=True)
        clips, texts = [], set()
        with tarfile.open(tar) as tf:
            for m in tf:
                fn = Path(m.name).name
                if not m.isfile() or fn not in rows or rows[fn]["text"] in texts:
                    continue
                y, sr = sf.read(io.BytesIO(tf.extractfile(m).read()))
                sf.write(str(out / code / fn), y, sr)
                texts.add(rows[fn]["text"])
                clips.append({"path": f"FLEURS/{code}/{fn}", "gender": rows[fn]["gender"], "text": rows[fn]["text"]})
                if len(clips) >= per_lang:
                    break
        meta[code] = {"language": name.replace("-", " ").lower(), "group": GROUP_OF[name], "clips": clips}
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f"{code}: {len(clips)} clips ({meta[code]['language']}, {meta[code]['group']})", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--per-lang", type=int, default=40)
    fetch(ap.parse_args().per_lang)
