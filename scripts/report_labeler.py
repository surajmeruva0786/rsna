"""Multilingual, negation-aware labeller mapping a free-text knee-MRI report to the
twelve competition targets.

Only 58 of the 4,407 training studies carry per-condition labels; the remaining 4,349
carry only the radiologist's narrative. This module turns those narratives into
training labels, which is what makes the other 98.7% of the training set usable.

Reports arrive in at least English, Spanish, German, Turkish, French, Portuguese and
Italian, so every pattern below is a multilingual alternation. Text is accent-folded
before matching, so a pattern written unaccented matches both `menisco` and `menisco`
spellings, `rotura` and `rotura`, `odem` and `odem`.

The labeller is deliberately rule-based rather than learned: with 58 gold examples
there is nothing to train a text model on, and the rules are auditable against those
58 (see `validate_labeler.py`).

Emits a probability rather than a hard bit: 1.0 for an asserted finding, 0.0 for an
explicitly negated one, and a calibrated prior for the "report never mentions it"
case, which is not the same as absence.
"""

from __future__ import annotations

import re
import unicodedata

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# --------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------

# Characters with no Unicode decomposition that still need folding. Greek final sigma
# is folded to medial sigma so a pattern need only spell one of them; the micro sign
# (which several Greek reports use in place of mu) is handled by NFKD below.
_EXTRA_FOLD = str.maketrans(
    {"ı": "i", "İ": "i", "ø": "o", "Ø": "o", "ß": "ss", "đ": "d", "ς": "σ"}
)


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_EXTRA_FOLD).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text)


# --------------------------------------------------------------------------------
# Shared vocabulary
# --------------------------------------------------------------------------------

# Words that cancel a finding mentioned in the same clause. Covers pre-posed negation
# (English/Spanish "no tear") and post-posed negation (Turkish "yirtik izlenmedi",
# German "Riss nicht nachweisbar"), which is why negation is searched clause-wide
# rather than only to the left of the match.
NEGATION = r"""(?:
      no\b | not\b | non\b | nor\b | without | absent | absence | negative\ for
    | free\ of | rules?\ out | ruled\ out | unremarkable | within\ normal
    | normal | intact | preserved | continuous | maintained
    | sin\b | ausencia | integr[oa] | conservad[oa] | preservad[oa] | indemne
    | no\ se\ (?:observa|identifica|aprecia|visualiza|evidencia|objetiva|demuestra)
    | no\ hay | no\ existe | descarta
    | kein[e]?[nrms]?\b | ohne\b | nicht\b | intakt | unauffaellig | unauffallig
    | regelrecht | frei\b | ausgeschlossen
    | yok\b | yoktur | izlenme(?:di|mistir) | saptanma(?:di|mistir)
    | goruelme(?:di)? | gorulme(?:di|mistir) | mevcut\ degil | dogal\b | saglam
    | pas\ de | absence | integre
    | senza | assenza | nessun[ao]?
    | sem\b | ausencia
    | geen\b | niet\b | zonder | normaal | onopvallend | ongestoord
    | δεν\b | χωρισ | ουδεν | εντοσ\ του\ φυσιολογικου | φυσιολογικ | ελευθερ
    | ακεραι | αναλλοιωτ | απουσια
)"""

# "possible/suspected" — treated as a weaker positive, not a negation.
HEDGE = r"(?:possibl|probabl|suspect|question|likely|may\ be|maybe|cannot\ exclude|no\ se\ puede\ descartar|posibl|probabl|sospech|dudos|verdacht|v\.\s?a\.|moeglich|moglich|fraglich|suephe|suphe|olabilir|ihtimal)"

# Pathology of a ligament or meniscus: a tear, in every language present.
TEAR = r"""(?:
      tear | tears | torn | rupture[ds]? | ruptur | disrupt | discontinu
    | avulsion | detach | complete\ (?:tear|disruption) | partial\ (?:tear|thickness\ tear)
    | rotura | ruptura | roto | rota | desgarro | desinsercion | lesion\ (?:completa|parcial)
    | riss | rupturiert | durchtrennt | kontinuitaetsunterbrechung | laesion
    | yirtik | yirtigi | rupture | rupturu | kopma
    | dechirure | lesion | rottura | lacerac
    | scheur | ruptuur | letsel | doorscheur
    | ρηξη | ρηξ[εη] | ρηγμα | διατομ | ασυνεχει | κακωσ
)"""

# Signal-abnormality phrasings that a radiologist uses to mean the same thing.
TEAR_SOFT = r"(?:in\ favou?r\ of\ tear|grade\ (?:iii|3)\ signal|reaching\ the\ (?:articular\ )?surface|extend(?:s|ing)\ (?:in)?to\ the\ (?:articular\ )?surface|signal\ (?:extending|reaching)\ the\ surface|senal\ que\ contacta|contacta\ con\ la\ superficie|alcanza\ la\ superficie)"

MEDIAL = r"(?:medial|intern[oa]|inner|innen|mediale[nrms]?|ic\b|iyan|interne|mediale|medial)"
LATERAL = r"(?:lateral|extern[oa]|outer|aussen|ausen|laterale[nrms]?|dis\b|externe|laterale)"


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.VERBOSE | re.IGNORECASE)


# --------------------------------------------------------------------------------
# Per-target anatomy patterns
# --------------------------------------------------------------------------------

ANATOMY = {
    "ACL": _rx(
        r"""(?:
              \bacl\b | anterior\ cruciate | ligamento\ cruzado\ anterior | \blca\b
            | cruzado\ anterior | vorder(?:es|en)\ kreuzband | \bvkb\b
            | on\ capraz\ bag | \bocb\b | ligament\ croise\ anterieur
            | legamento\ crociato\ anteriore | \blcae?\b
            | voorste\ kruisband | \bvkb\b
            | προσθι\w*\ χιαστ | χιαστ\w*\ προσθι
        )"""
    ),
    "MCL": _rx(
        r"""(?:
              \bmcl\b | medial\ collateral | tibial\ collateral
            | ligamento\ (?:colateral|lateral)\ (?:medial|interno) | \blli\b | \blcm\b
            | innenband | mediale[sn]?\ (?:kollateral|seiten)band
            | ic\ yan\ bag | medial\ kollateral
            | ligament\ collateral\ (?:medial|interne) | legamento\ collaterale\ mediale
            | mediale\ collaterale\ band | binnenband
            | εσω\ πλαγι | πλαγι\w*\ εσω
        )"""
    ),
    "Medial Meniscus": _rx(
        r"""(?:
              medial\ meniscus | meniscus\ medial | menisco\ (?:medial|interno)
            | innenmeniskus | mediale[rn]?\ meniskus | ic\ menisku[sc]
            | medial\ menisku[sc] | menisque\ (?:medial|interne) | menisco\ mediale
            | \bmm\b(?=\ (?:tear|rotura))
            | mediale\ meniscus | meniscus\ mediaal | binnenmeniscus
            | εσω\ μηνισκ | μηνισκ\w*\ εσω | εσωτερικ\w*\ μηνισκ
        )"""
    ),
    "Lateral Meniscus": _rx(
        r"""(?:
              lateral\ meniscus | meniscus\ lateral | menisco\ (?:lateral|externo)
            | aussenmeniskus | ausenmeniskus | laterale[rn]?\ meniskus
            | dis\ menisku[sc] | lateral\ menisku[sc] | menisque\ (?:lateral|externe)
            | menisco\ laterale
            | laterale\ meniscus | meniscus\ lateraal | buitenmeniscus
            | εξω\ μηνισκ | μηνισκ\w*\ εξω | εξωτερικ\w*\ μηνισκ
        )"""
    ),
}

# Osteoarthritis / cartilage-loss vocabulary. The competition's OA targets are
# compartment-specific, so these must be paired with a laterality or compartment cue.
OA_TERM = r"""(?:
      osteoarthrit | osteoarthros | degenerative\ (?:change|disease|arthropathy)
    | cartilage\ (?:loss|thinning|defect|wear|degeneration) | chondral\ (?:loss|thinning|defect|wear)
    | chondromalacia | joint\ space\ narrowing | osteophyt | spur
    | artrosis | artrosic | osteoartrosis | gonartrosis | condropat | condromalacia
    | desgaste\ (?:cartilaginoso|condral) | pinzamiento\ (?:articular|femorotibial)
    | adelgazamiento\ (?:del\ )?cartilago
    | arthrose | chondropathie | chondromalazie | knorpel(?:verschmaelerung|schaden|defekt|glatze|abnutzung)
    | gelenkspaltverschmaelerung | osteophyt
    | osteoartrit | kondropati | kikirdak\ (?:kaybi|incelme|hasari) | dejeneratif
    | arthrose | condropatia | artrosi
    | kraakbeen(?:lijden|verlies|schade|defect) | osteofyt | artrose | chondropathie
    | verdund\ voorkomen\ kraakbeen | gewrichtslijden | gewrichtsspleetversmalling
    | αρθριτ | αρθρωσ | αρθροπαθει | οστεοφυτ | εκφυλιστικ
    | χονδροπαθει | χονδρομαλακ | αραιωσ\w*\ χονδρ
    | εξαλειψη\ του\ αρθρικου\ χονδρου | βλαβ\w*\ του\ χονδρ
    | στενωσ\w*\ του\ μεσαρθριου
)"""

COMPARTMENT = {
    "Medial OA": _rx(
        rf"""(?:
              (?:medial|internal)\ (?:tibiofemoral|femorotibial|femoro-tibial)?\s*compartment
            | (?:tibiofemoral|femorotibial)\ (?:medial|internal)
            | femorotibial\ (?:medial|intern[oa]) | (?:medial|intern[oa])\ femorotibial
            | medial\ (?:femoral\ condyle|tibial\ plateau)
            | condilo\ femoral\ (?:medial|interno) | platillo\ tibial\ (?:medial|interno)
            | mediale[sn]?\ (?:kompartiment|gelenkkompartiment) | innere[sn]?\ kompartiment
            | medial\ kompartman | ic\ kompartman
            | mediaal\ femorotibiaal | mediale\ (?:femurcondyl|tibiaplateau)
            | εσω\ διαμερισμα | εσω\ κνημιαι | εσω\ μηριαι | μηροκνημιαι\w*\ εσω
            | εσω\ κονδυλ
        )"""
    ),
    "Lateral OA": _rx(
        rf"""(?:
              (?:lateral|external)\ (?:tibiofemoral|femorotibial|femoro-tibial)?\s*compartment
            | (?:tibiofemoral|femorotibial)\ (?:lateral|external)
            | femorotibial\ (?:lateral|extern[oa]) | (?:lateral|extern[oa])\ femorotibial
            | lateral\ (?:femoral\ condyle|tibial\ plateau)
            | condilo\ femoral\ (?:lateral|externo) | platillo\ tibial\ (?:lateral|externo)
            | laterale[sn]?\ (?:kompartiment|gelenkkompartiment) | ausere[sn]?\ kompartiment
            | lateral\ kompartman | dis\ kompartman
            | lateraal\ femorotibiaal | laterale\ (?:femurcondyl|tibiaplateau)
            | εξω\ διαμερισμα | εξω\ κνημιαι | εξω\ μηριαι | μηροκνημιαι\w*\ εξω
            | εξω\ κονδυλ
        )"""
    ),
    "PF OA": _rx(
        r"""(?:
              patell?ofemoral | femoropatell?ar | femoro-patell?ar | patello-femoral
            | retropatell | pre-?patell?ar\ cartilage | trochlea | troclea | trochlear
            | patellar\ (?:facet|cartilage) | rotulian[oa] | patelar
            | femoropatelar | femorotroclear
            | patellagleitlager | patellofemoral | retropatellar
            | patellofemoral\ eklem | patellar\ kikirdak
            | femoropatellair | patellofemoraal | retropatellair | kraakbeen\ patella
            | επιγονατιδομηριαι | μηροεπιγονατιδικ | οπισθοεπιγονατιδ | επιγονατιδ
            | τροχιλ
        )"""
    ),
}

# Compartment-agnostic OA phrasings that imply several compartments at once.
GLOBAL_OA = {
    "tri": _rx(r"(?:tricompartment|tricompartiment|tri-compartment|pangonartrosis|dreikompartiment)"),
    "bi": _rx(r"(?:bicompartment|bicompartiment|bi-compartment|zweikompartiment)"),
    "gon": _rx(r"(?:gonartrosis|gonarthrose|gonartroz|knee\ osteoarthrit)"),
}

# Standalone findings: the term itself is the finding, no anatomy pairing needed.
STANDALONE = {
    "Effusion": _rx(
        r"""(?:
              (?:joint\ )?effusion | intra-?articular\ fluid | joint\ fluid
            | fluid\ in\ the\ (?:joint|suprapatellar) | distended\ suprapatellar
            | derrame | hidrartrosis | liquido\ (?:articular|intraarticular)
            | aumento\ de\ liquido | ergu(?:ss|ess) | gelenkerguss | gelenkfluessigkeit
            | erguss | efuzyon | eklem\w*\ (?:ici\ )?(?:sivi|mayi) | epanchement | versamento
            | derrame\ articular | sivi\ miktari\w*\ (?:artmis|fazla)
            | gewrichtsvocht | hydrops | vocht\ in\ het\ gewricht | vochtcollectie
            | (?:ενδαρθρικ|ενδοαρθρικ|αρθρικ)\w*\ συλλογη | συλλογη\ υγρου
            | υπερβολικ\w*\ υγρ | υδραρθρ
        )"""
    ),
    "Synovitis": _rx(
        r"""(?:
              synovit | synovial\ (?:thickening|proliferation|hypertroph|enhancement|reaction)
            | thicken(?:ed|ing)\ synovi | pannus
            | sinovit | engrosamiento\ sinovial | proliferacion\ sinovial | hipertrofia\ sinovial
            | synovialit | synoviale?\ (?:verdickung|proliferation|hypertrophie|reizung)
            | sinovyal\ (?:kalinlasma|proliferasyon) | synovite | sinovite
            | synoviale?\ (?:verdikking|proliferatie|woekering) | verdikking\w*\ van\ (?:het\ )?synovium
            | υμενιτ | αρθρικ\w*\ υμεν | υμενικ\w*\ (?:παχυνσ|υπερτροφ|πολλαπλασιασ)
            | παχυνση\ του\ αρθρικου\ υμενα | αρθριτιδα\ υμεν
        )"""
    ),
    "Baker's": _rx(
        r"""(?:
              baker | popliteal\ cyst | poplitea?l?\ bursa\ (?:cyst|distension)
            | gastrocnemio-?semimembranosus\ bursa
            | quiste\ (?:de\ baker|poplite[oa]) | quistes?\ poplite
            | bakerzyste | baker-?zyste | poplitealzyste | poplitea\ zyste
            | poplitea\ kisti | baker\ kisti | kyste\ (?:de\ baker|poplite)
            | cisti\ di\ baker
            | bakercyste | popliteale\ cyste | baker\ cyste
            | κυστη\ (?:του\ )?baker | κυστη\ μπεικερ | ιγνυακ\w*\ κυστ
        )"""
    ),
    "Contusion": _rx(
        r"""(?:
              (?:bone|marrow)\ (?:contusion|bruise|oedema|edema)
            | bone\ marrow\ (?:oedema|edema|signal\ (?:change|alteration))
            | trabecular\ (?:oedema|edema|microfracture)
            | contusion\ (?:osea|trabecular) | edema\ (?:oseo|de\ medula\ osea|medular)
            | edema\ subcondral | contusion\ osea
            | knochenmark(?:oedem|odem|s?oedem|sodem) | knochenoedem | knochenodem
            | subchondrale[sn]?\ (?:oedem|odem) | kontusion | bone\ bruise
            | kemik\ (?:iligi\ )?odem | subkondral\ odem | contusion\ osseuse
            | oedeme\ (?:osseux|medullaire) | edema\ osseo
            | kontuzyon | kemik\ kontuzyon
            | botoedeem | beenmergoedeem | subchondraal\ oedeem | botcontusie
            | οστικ\w*\ (?:μωλωπ|θλασ) | οιδημα\ (?:του\ )?(?:οστικου\ )?μυελου
            | οστεομυελικ\w*\ οιδημα | μυελικ\w*\ οιδημα | υποχονδρι\w*\ οιδημα
        )"""
    ),
    "Fracture": _rx(
        r"""(?:
              fracture[ds]? | fractured | fissure\ line | cortical\ break
            | fractura | fisura\ osea
            | fraktur | frakturen | knochenbruch | infraktion
            | kirik | fissure | frattura | fratura
            | fractuur | botbreuk
            | καταγμα | καταγματ | οστικ\w*\ ρωγμ
        )"""
    ),
}

# Soft evidence: findings that raise the odds of a target without asserting it, scored
# between "never mentioned" and "asserted" so they rank correctly under AUC.
#
# Two motivations. Synovitis is frequently not named outright but reported through its
# companions — bursal distension, Hoffa fat-pad oedema, plica irritation — and the gold
# labels mark those studies positive. Conversely, isolated *subchondral* oedema is
# usually the reactive marrow change beside an arthritic compartment rather than a
# traumatic bruise, so it belongs here rather than in Contusion's assertive tier.
SOFT_SCORE = 0.45

SOFT = {
    "Synovitis": _rx(
        r"""(?:
              bursit | suprapatellar\ (?:bursa\ )?(?:distension|distended|effusion)
            | distended\ suprapatellar | hoffit | hoffa\ (?:fat\ pad\ )?(?:oedema|edema|impingement)
            | fat\ pad\ (?:oedema|edema|impingement|stranding) | plica\ (?:syndrome|thickening|irritation)
            | arthritis | arthrit
            | bursitis | almohadilla\ grasa\ de\ hoffa | pinzamiento\ de\ la\ almohadilla
            | sinovial | grasa\ de\ hoffa
            | bursitis | schleimbeutel | hoffa-?koerper | reizerguss | reizzustand
            | bursit | hoffa\ yag\ yastigi
            | bursitis | slijmbeurs
            | θυλακιτ | ορογονοθυλακιτ | λιπωδ\w*\ σωματ\w*\ hoffa | υμεν
        )"""
    ),
    "Contusion": _rx(
        r"""(?:
              subchondral\ (?:oedema|edema) | edema\ subcondral | subchondrale[sn]?\ (?:oedem|odem)
            | subkondral\ odem | subchondraal\ oedeem | υποχονδρι\w*\ οιδημα
            | marrow\ signal\ (?:change|alteration|abnormality) | alteracion\ de\ senal\ (?:de\ la\ )?medula
        )"""
    ),
}

# `Contusion` and `Fracture` overlap in phrasing; an osteochondral *fracture* should
# not be swallowed by the contusion rule and vice versa. These re-assert Fracture.
FRACTURE_STRONG = _rx(
    r"(?:osteochondral\ fracture|avulsion\ fracture|insufficiency\ fracture|impaction\ fracture|segond|stress\ fracture|fractura\ (?:osteocondral|por\ (?:estres|insuficiencia))|abrissfraktur|impressionsfraktur)"
)

# Phrases that mark a sentence as historical / not a current finding.
HISTORY = _rx(
    r"(?:post-?operative|postoperator|status\ post|s/?p\ |prior\ |previous(?:ly)?|history\ of|antecedente|zustand\ nach|z\.\s?n\.|opere|ameliyat|reconstruct|plasty|plastia|graft|injerto|transplant|meniscectom|meniscectomia|resected|resecc)"
)


# --------------------------------------------------------------------------------
# Clause segmentation
# --------------------------------------------------------------------------------

# Only strong boundaries split a segment. Notably `:` does NOT split: structured
# reports write "Medial meniscus: complete tearing of the body", and splitting there
# would strand the anatomy from its pathology. For the same reason "and", "with" and
# commas are not boundaries — "tear at the body and posterior horn of the medial
# meniscus" has to survive intact.
_SEGMENT_SPLIT = re.compile(r"(?:[.;!?\n>]|\s--\s|\s-\s|•)+")

# How far apart an anatomy mention and its pathology may sit and still be one finding.
PAIR_WINDOW = 120
# Negation is searched in a tight span around the matched pair, not segment-wide, so
# that "tear of the medial meniscus and the lateral meniscus is intact" negates only
# the lateral half.
NEG_BEFORE = 45
NEG_AFTER = 30

_NEGATION_RX = re.compile(NEGATION, re.VERBOSE | re.IGNORECASE)
_HEDGE_RX = re.compile(HEDGE, re.IGNORECASE)
_TEAR_RX = re.compile(TEAR, re.VERBOSE | re.IGNORECASE)
_TEAR_SOFT_RX = re.compile(TEAR_SOFT, re.VERBOSE | re.IGNORECASE)
_OA_RX = re.compile(OA_TERM, re.VERBOSE | re.IGNORECASE)


def segments(text: str) -> list[str]:
    """Split normalised report text into sentence-like segments."""
    return [s.strip() for s in _SEGMENT_SPLIT.split(text) if s and s.strip()]


def _nearest(rx: re.Pattern, seg: str, anchor: re.Match) -> re.Match | None:
    """The match of `rx` in `seg` closest to `anchor`, within PAIR_WINDOW."""
    best, best_d = None, PAIR_WINDOW + 1
    for m in rx.finditer(seg):
        d = 0 if m.start() < anchor.end() and m.end() > anchor.start() else min(
            abs(m.start() - anchor.end()), abs(anchor.start() - m.end())
        )
        if d < best_d:
            best, best_d = m, d
    return best if best_d <= PAIR_WINDOW else None


def _span_negated(seg: str, a: re.Match, b: re.Match | None) -> bool:
    """Is the anatomy/pathology pair negated?

    Negation is resolved over a tight span around the pair rather than to the left of
    it, because several represented languages post-pose the negation ("yirtik
    izlenmedi", "Meniskusriss nicht nachweisbar", "menisco integro").
    """
    lo = min(a.start(), b.start() if b else a.start()) - NEG_BEFORE
    hi = max(a.end(), b.end() if b else a.end()) + NEG_AFTER
    return _NEGATION_RX.search(seg[max(0, lo) : hi]) is not None


def _span_hedged(seg: str, a: re.Match, b: re.Match | None) -> bool:
    lo = min(a.start(), b.start() if b else a.start()) - NEG_BEFORE
    hi = max(a.end(), b.end() if b else a.end()) + NEG_AFTER
    return _HEDGE_RX.search(seg[max(0, lo) : hi]) is not None


# --------------------------------------------------------------------------------
# Labelling
# --------------------------------------------------------------------------------

# Score assigned when the report asserts / hedges a finding. `MISSING` is used when
# the report never mentions the finding at all, which is weaker evidence of absence
# than an explicit negation and so is not driven to 0.
ASSERTED = 1.0
HEDGED = 0.65
NEGATED = 0.02
MISSING = 0.06


def label_report(report: str) -> dict[str, float]:
    """Score one report against the twelve targets. Returns a value in [0, 1] each."""
    text = normalize(report)
    segs = segments(text)

    # Positive assertions and explicit negations are tracked separately rather than
    # max-reduced together: a report that says "ACL intact" is stronger evidence of
    # absence than one that never mentions the ACL, and a max over both would erase
    # that difference.
    pos: dict[str, float] = {t: 0.0 for t in TARGETS}
    neg: dict[str, bool] = {t: False for t in TARGETS}

    def bump(target: str, value: float) -> None:
        if value == NEGATED:
            neg[target] = True
        elif value > pos[target]:
            pos[target] = value

    for seg in segs:
        historical = HISTORY.search(seg) is not None

        # Ligament and meniscus targets need anatomy AND a nearby tear; "ACL is
        # intact" matches anatomy alone and must not fire.
        for target, anat in ANATOMY.items():
            for a in anat.finditer(seg):
                tear = _nearest(_TEAR_RX, seg, a) or _nearest(_TEAR_SOFT_RX, seg, a)
                if tear is None:
                    continue
                if _span_negated(seg, a, tear):
                    bump(target, NEGATED)
                elif historical:
                    # A reconstructed ACL still marks the study as having had it.
                    bump(target, HEDGED)
                else:
                    bump(target, HEDGED if _span_hedged(seg, a, tear) else ASSERTED)
                break

        # Compartment-specific osteoarthritis: an OA term paired with a compartment.
        for oa in _OA_RX.finditer(seg):
            fired = False
            for target, comp in COMPARTMENT.items():
                c = _nearest(comp, seg, oa)
                if c is None:
                    continue
                fired = True
                if _span_negated(seg, oa, c):
                    bump(target, NEGATED)
                else:
                    bump(target, HEDGED if _span_hedged(seg, oa, c) else ASSERTED)
            if fired or _span_negated(seg, oa, None):
                continue
            value = HEDGED if _span_hedged(seg, oa, None) else ASSERTED
            if GLOBAL_OA["tri"].search(seg):
                for target in ("Medial OA", "Lateral OA", "PF OA"):
                    bump(target, value)
            elif GLOBAL_OA["bi"].search(seg):
                for target in ("Medial OA", "PF OA"):
                    bump(target, value)
            elif GLOBAL_OA["gon"].search(seg):
                # Unqualified gonarthrosis is medial-dominant in practice.
                bump("Medial OA", value * 0.8)
                bump("PF OA", value * 0.5)
                bump("Lateral OA", value * 0.4)

        # Standalone findings: the term is the finding, no anatomy pairing needed.
        for target, rx in STANDALONE.items():
            for m in rx.finditer(seg):
                if _span_negated(seg, m, None):
                    bump(target, NEGATED)
                else:
                    bump(target, HEDGED if _span_hedged(seg, m, None) else ASSERTED)
                    break

        # Soft evidence never negates, only raises a floor.
        for target, rx in SOFT.items():
            for m in rx.finditer(seg):
                if not _span_negated(seg, m, None):
                    bump(target, SOFT_SCORE)
                    break

        for m in FRACTURE_STRONG.finditer(seg):
            if not _span_negated(seg, m, None):
                bump("Fracture", ASSERTED)

    return {
        t: pos[t] if pos[t] > 0.0 else (NEGATED if neg[t] else MISSING)
        for t in TARGETS
    }
