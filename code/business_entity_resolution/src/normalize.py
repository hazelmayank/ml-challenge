"""Text normalisation for business names and addresses (vectorised with polars).

Produces, per record:
  name_norm   lowercase ASCII name, punctuation/junk removed
  name_core   name_norm without legal suffixes / generic filler words
  name_compact name_core with spaces removed (matches "acme.com" to "Acme Inc")
  addr_norm   lowercase ASCII address with abbreviations/states canonicalised
  addr_nums   space-joined numeric tokens of the address (leading zeros dropped)
  addr_ids    compound address identifiers kept whole ("5/11/2", "24/309", "1056c")
  hnum        first address identifier (usually the house / door number)
  legal       canonical legal forms + credentials in the name ("ltd pvt", "llc", "dmd")
  script      dominant non-Latin script of the raw name ("latin", "indic")

Indic-script names are translated word by word with WORK/indic_dict.json (learned from
training pairs by learn_indic.py); unknown words fall back to unidecode.
"""
import json
import re

import polars as pl
from unidecode import unidecode

from config import WORK

LEGAL_CANON = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "pvt": "pvt",
    "private": "pvt", "public": "public", "llp": "llp", "lp": "lp", "plc": "plc", "pc": "pc",
    "pllc": "pllc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "gmbh": "gmbh", "md": "md", "dmd": "dmd", "dds": "dds", "od": "od",
    "cpa": "cpa", "esq": "esq", "phd": "phd", "dvm": "dvm", "jr": "jr", "sr": "sr",
    "pa": "pa", "ltda": "ltd", "snc": "snc",
}
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "lp", "plc", "pc", "pllc", "sarl", "sas", "sa", "eurl", "sasu",
    "gmbh", "the", "and", "of", "dba", "d", "b", "a", "center", "centre", "services",
    "service", "partners", "group", "esq", "smt", "dr", "mr", "mrs", "shri", "sri", "et",
    "cie", "l", "le", "la", "les", "de", "du", "des",
}

STREET = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "rd": "road",
    "dr": "drive", "blvd": "boulevard", "ln": "lane", "hwy": "highway", "pkwy": "parkway",
    "ter": "terrace", "terr": "terrace", "pl": "place", "sq": "square", "cir": "circle",
    "ct": "court", "trl": "trail", "fl": "floor", "flr": "floor", "apt": "apartment",
    "ste": "suite", "bldg": "building", "mt": "mount", "ft": "fort", "hyd": "hyderabad",
    "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata", "bengaluru": "bangalore",
    "no": "", "h": "", "door": "", "doro": "", "plot": "", "shop": "", "null": "",
    "saint": "street", "twp": "township", "townshiip": "township", "ciyt": "city",
    "nagr": "nagar", "po": "", "box": "", "unit": "", "suite": "", "rue": "rue",
    # French
    "bd": "boulevard", "bld": "boulevard", "che": "chemin", "chem": "chemin",
    "imp": "impasse", "fg": "faubourg", "fbg": "faubourg", "rte": "route", "bis": "",
}
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in",
    "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "or", "orissa": "or", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "chandigarh": "ch", "puducherry": "py",
    "jammu and kashmir": "jk",
}
# States are only canonicalised in addresses; applied as whole-phrase replacements.
STATE_PATTERNS = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))

# Words the data generator injects into S2/S3 names, found per country as words far more
# frequent in S2/S3 than in S1 (France measured on the test set: it has no training data).
# Applied only to that country's records; other countries get the generic LEGAL list.
FILLER = {
    "France": {"participations", "holding", "snc", "distribution", "associes",
               "international", "developpement", "groupe", "et", "as"},
    "US": {"formerly", "midtown", "northside", "eastgate", "greater", "southside",
           "riverside", "westgate", "lakeside", "as", "aka"},
    "India": {"m", "s", "overseas", "infratech", "as", "aka"},
}
# French abbreviations S2/S3 use and S1 spells out ("R." is in 25% of French S2/S3 records),
# and departments that S2/S3 use in place of S1's region.
FR_STREET = {"r": "rue", "ch": "chemin", "crs": "cours", "q": "quai", "all": "allee",
             "appt": "appartement", "app": "appartement", "bat": "batiment",
             "res": "residence", "ste": "sainte"}
FR_REGIONS = sorted({
    "loire atlantique": "pays de la loire", "gironde": "nouvelle aquitaine",
    "nord": "hauts de france", "pas de calais": "hauts de france",
}.items(), key=lambda kv: -len(kv[0]))
# Indian state names written in Indic script inside S2/S3 addresses (~2M fields); plain
# transliteration turns them into "mhaaraassttr", "dillii", ... which match nothing.
ADDR_INDIC = {
    "महाराष्ट्र": "Maharashtra", "दिल्ली": "Delhi", "उत्तर प्रदेश": "Uttar Pradesh",
    "ಕರ್ನಾಟಕ": "Karnataka", "தமிழ்நாடு": "Tamil Nadu", "ગુજરાત": "Gujarat",
    "পশ্চিমবঙ্গ": "West Bengal", "తెలంగాణ": "Telangana", "हरियाणा": "Haryana",
    "राजस्थान": "Rajasthan", "കേരളം": "Kerala", "बिहार": "Bihar",
    "मध्य प्रदेश": "Madhya Pradesh", "ఆంధ్రప్రదేశ్": "Andhra Pradesh", "ਪੰਜਾਬ": "Punjab",
    "ଓଡ଼ିଶା": "Odisha",
}

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_INDIC = re.compile(r"[ऀ-෿]")


_DICT_PATH = WORK / "indic_dict.json"
INDIC_DICT = (json.loads(_DICT_PATH.read_text(encoding="utf-8"))
              if _DICT_PATH.exists() else {})


def _translate(text: str) -> str:
    return unidecode(" ".join(INDIC_DICT.get(w, w) for w in text.split()))


def to_ascii(s: pl.Series, translate=False) -> pl.Series:
    """unidecode only the rows that need it (a few % of the data); optionally translate
    Indic words with the learned dictionary first."""
    mask = s.str.contains(r"[^\x00-\x7f]")
    if not mask.any():
        return s
    fn = _translate if translate and INDIC_DICT else unidecode
    fixed = s.filter(mask).map_elements(fn, return_dtype=pl.String)
    return s.scatter(mask.arg_true(), fixed)


def _replace_words(expr: pl.Expr, mapping: dict) -> pl.Expr:
    words = list(mapping)
    return expr.str.replace_many([f" {w} " for w in words],
                                 [f" {mapping[w]} " if mapping[w] else " " for w in words])


def _by_country(base: pl.Expr, variants: dict) -> pl.Expr:
    """base, replaced by variants[country](base) for the countries listed."""
    out = base
    for country, fn in variants.items():
        out = pl.when(pl.col("country") == country).then(fn(base)).otherwise(out)
    return out


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    """df has entity_id, business_name, business_address, country."""
    name_raw = df["business_name"]
    addr_raw = (df["business_address"]
                .str.replace_many(list(ADDR_INDIC), list(ADDR_INDIC.values()))
                .str.replace_all("Â ", " ")      # mojibake non-breaking space
                .str.replace_all(r"[°º]", "o"))              # N° / Nº -> "No" (dropped)
    df = df.with_columns(
        script=pl.when(name_raw.str.contains(r"[ऀ-෿]")).then(pl.lit("indic"))
        .otherwise(pl.lit("latin")),
        is_domain=name_raw.str.to_lowercase().str.contains(r"^\S+\.(com|net|org|in|co|fr|us)$"),
        addr_missing=addr_raw.str.strip_chars().str.len_chars() == 0,
        name_ascii=to_ascii(name_raw, translate=True),
        addr_ascii=to_ascii(addr_raw),
    )

    name = (pl.col("name_ascii").str.to_lowercase()
            .str.replace(r"\.(com|net|org|in|co|fr|us)$", "")
            .str.replace_all(r"\bid\s*:?\s*\d+", " ")          # "(ID: 34016)" junk
            .str.replace_all(r"\b(?:[dlj]|qu)['`]\s*", "")      # French elision d'/l'/qu'
            .str.replace_all(r"[&+]", " and ")
            .str.replace_all(r"\.", "")
            .str.replace_all(r"\d{6,}", " ")
            .str.replace_all(r"[^a-z0-9]+", " "))
    name = pl.concat_str(pl.lit(" "), name, pl.lit(" "))
    df = df.with_columns(name_norm=name.str.replace_all(r"\s+", " ").str.strip_chars())
    core = pl.concat_str(pl.lit(" "), pl.col("name_norm").str.replace_all(" ", "  "), pl.lit(" "))
    core = _replace_words(core, {w: "" for w in LEGAL})
    core = _by_country(core, {c: (lambda e, ws=ws: _replace_words(e, {w: "" for w in ws}))
                              for c, ws in FILLER.items()})
    df = df.with_columns(name_core=core.str.replace_all(r"\s+", " ").str.strip_chars())
    df = df.with_columns(
        name_core=pl.when(pl.col("name_core") == "").then(pl.col("name_norm"))
        .otherwise(pl.col("name_core")))
    df = df.with_columns(
        name_compact=pl.col("name_core").str.replace_all(" ", ""),
        legal=pl.col("name_norm").str.split(" ")
        .list.eval(pl.element().replace_strict(LEGAL_CANON, default=None).drop_nulls()
                   .unique().sort())
        .list.join(" "),
    )

    addr_lower = (pl.col("addr_ascii").str.to_lowercase()
                  .str.replace_all(r"<?null>?", " ")
                  .str.replace_all(r"\b(?:[dlj]|qu)['`]\s*", "")
                  .str.replace_all(r"(\d+)(?:eme|e|er|ere)\b", "$1")
                  .str.replace_all(r"(\d+)(st|nd|rd|th)\b", "$1"))
    df = df.with_columns(
        addr_ids=addr_lower.str.extract_all(r"[0-9a-z]*\d[0-9a-z]*(?:[/-][0-9a-z]*\d[0-9a-z]*)*")
        .list.eval(pl.element().str.replace_all(r"(^|[/-])0+(\d)", "$1$2"))
        .list.join(" "))
    df = df.with_columns(hnum=pl.col("addr_ids").str.extract(r"^(\S+)"))
    addr = addr_lower.str.replace_all(r"[^a-z0-9]+", " ")
    addr = pl.concat_str(pl.lit(" "), addr.str.replace_all(" ", "  "), pl.lit(" "))
    addr = _replace_words(addr, ORDINAL_WORDS)
    addr = _by_country(addr, {"France": lambda e: _replace_words(e, FR_STREET)})
    addr = _replace_words(addr, STREET)
    addr = addr.str.replace_all(r"\s+", " ")
    addr = _replace_words(addr, dict(STATE_PATTERNS))
    addr = _by_country(addr, {"France": lambda e: _replace_words(e, dict(FR_REGIONS))})
    addr = _replace_words(addr, {"pmb": "", "cdp": ""})     # US S2/S3-only tokens
    addr = addr.str.replace_all(r"\b0+(\d)", "$1")
    df = df.with_columns(addr_norm=addr.str.replace_all(r"\s+", " ").str.strip_chars())
    df = df.with_columns(
        addr_nums=pl.col("addr_norm").str.extract_all(r"\d+").list.join(" "))
    return df.drop("name_ascii", "addr_ascii")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo = pl.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S3-1", "S2-2", "S3-2", "S2-3"],
        "business_name": ["Maure Williams Colombier Inc", ">> CALDEON NOVA LLC",
                          "maurewilliamscolombier.com", "एसएस फूड प्राइवेट लिमिटेड",
                          "[Inc] AP Hospitality", "Chordia + Pagnters - 7306204978"],
        "business_address": ["85 Wayne Avenue, Ticonderoga, NY",
                             "MO, KIMBERLING CITY, 1 GREENBRIER DRIVE",
                             "Wayne Ave, Ticonderoga Townshiip, New York",
                             "AF-0684, Uttar Pradesh, GHAZIABAD, 9487203",
                             "20085-20089 Us 23, Circleville, Ohio",
                             "KANSAS CITY, MO, 630 45ND TERRACE, null"],
        "country": ["US", "US", "US", "India", "US", "US"],
    })
    with pl.Config(tbl_width_chars=250, fmt_str_lengths=60, tbl_cols=-1):
        print(normalize(demo).drop("business_name", "business_address", "country"))
