from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

@dataclass(frozen=True)
class Benchmark:
    """
    A benchmark/proficiency/achievement cutpoint scheme.
    cutpoints:
      - strictly increasing lower bounds separating bands.
      - interpretation depends on kind (IEA benchmarks, OECD proficiency levels, NAEP achievement levels).
    """
    key: str
    kind: str  # e.g. "iea_international_benchmarks"; "oecd_proficiency_levels"; "naep_achievement_levels"
    cutpoints: Tuple[float, ...]
    labels: Tuple[str, ...]
    note: str = ""


def _as_tuple(xs: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _validate(b: Benchmark) -> Benchmark:
    cps = b.cutpoints
    if any(cps[i] >= cps[i + 1] for i in range(len(cps) - 1)):
        raise ValueError(f"Benchmark {b.key}: cutpoints must be strictly increasing, got {cps}")
    return b

BENCHMARKS: Dict[str, Benchmark] = {}
ALIASES: Dict[str, str] = {}

def register(b: Benchmark, *, aliases: Optional[Iterable[str]] = None) -> None:
    b = _validate(b)
    if b.key in BENCHMARKS:
        raise KeyError(f"Benchmark already registered: {b.key}")
    BENCHMARKS[b.key] = b
    if aliases:
        for a in aliases:
            if a in ALIASES or a in BENCHMARKS:
                raise KeyError(f"Alias already used: {a}")
            ALIASES[a] = b.key


def resolve_key(key: str) -> str:
    if key in BENCHMARKS:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise KeyError(f"Unknown benchmark key: {key}")


def get_benchmark(key: str) -> Benchmark:
    return BENCHMARKS[resolve_key(key)]


def get_cutpoints(key: str) -> Tuple[float, ...]:
    return get_benchmark(key).cutpoints


def list_benchmarks() -> Tuple[str, ...]:
    keys = sorted(BENCHMARKS.keys())
    return tuple(keys)


register(
    Benchmark(
        key="iea:benchmarks:400_475_550_625",
        kind="iea_international_benchmarks",
        cutpoints=_as_tuple([400, 475, 550, 625]),
        labels=("low", "intermediate", "high", "advanced"),
        note="IEA international benchmarks (TIMSS/PIRLS).",
    ),
    aliases=[
        "timss",
        "pirls",
        "timss:benchmarks",
        "pirls:benchmarks",
        # cycles (same cutpoints; kept for user reproducibility)
        "timss2011",
        "timss2015",
        "timss2019",
        "timss2023",
        "pirls2016",
        "pirls2021",
    ],
)


register(
    Benchmark(
        key="piaac:litnum:176_226_276_326_376",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([176, 226, 276, 326, 376]),
        labels=("level_1", "level_2", "level_3", "level_4", "level_5"),
        note="PIAAC literacy/numeracy proficiency thresholds.",
    ),
    aliases=[
        "piaac",
        "piaac:literacy",
        "piaac:numeracy",
    ],
)

# PISA Reading
register(
    Benchmark(
        key="pisa2012:reading",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([262.04, 334.75, 407.47, 480.18, 552.89, 625.61, 698.32]),
        labels=("1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2012 reading proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2015:reading",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([262.04, 334.75, 407.47, 480.18, 552.89, 625.61, 698.32]),
        labels=("1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2015 reading proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2018:reading",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([189.33, 262.04, 334.75, 407.47, 480.18, 552.89, 625.61, 698.32]),
        labels=("1c", "1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2018 reading proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2022:reading",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([189.33, 262.04, 334.75, 407.47, 480.18, 552.89, 625.61, 698.32]),
        labels=("1c", "1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2022 reading proficiency cutpoints (NCES).",
    )
)

# PISA Science
register(
    Benchmark(
        key="pisa2012:science",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([334.94, 409.54, 484.14, 558.73, 633.33, 707.93]),
        labels=("1", "2", "3", "4", "5", "6"),
        note="PISA 2012 science proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2015:science",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([260.54, 334.94, 409.54, 484.14, 558.73, 633.33, 707.93]),
        labels=("1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2015 science proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2018:science",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([260.54, 334.94, 409.54, 484.14, 558.73, 633.33, 707.93]),
        labels=("1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2018 science proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2022:science",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([260.54, 334.94, 409.54, 484.14, 558.73, 633.33, 707.93]),
        labels=("1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2022 science proficiency cutpoints (NCES).",
    )
)

# PISA Math
register(
    Benchmark(
        key="pisa2012:math",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([357.77, 420.07, 482.38, 544.68, 606.99, 669.30]),
        labels=("1", "2", "3", "4", "5", "6"),
        note="PISA 2012 mathematics proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2015:math",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([357.77, 420.07, 482.38, 544.68, 606.99, 669.30]),
        labels=("1", "2", "3", "4", "5", "6"),
        note="PISA 2015 mathematics proficiency cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2018:math",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([357.77, 420.07, 482.38, 544.68, 606.99, 669.30]),
        labels=("1", "2", "3", "4", "5", "6"),
        note="PISA 2018 mathematics proficiency cutpoints (NCES; note indicates single level 1).",
    )
)
register(
    Benchmark(
        key="pisa2022:math",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([233.17, 295.47, 357.77, 420.07, 482.38, 544.68, 606.99, 669.30]),
        labels=("1c", "1b", "1a", "2", "3", "4", "5", "6"),
        note="PISA 2022 mathematics proficiency cutpoints (NCES).",
    )
)

# PISA Financial literacy 
register(
    Benchmark(
        key="pisa2018:financial",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([325.57, 400.33, 475.10, 549.86, 624.63]),
        labels=("1", "2", "3", "4", "5"),
        note="PISA 2018 financial literacy cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="pisa2022:financial",
        kind="oecd_proficiency_levels",
        cutpoints=_as_tuple([325.57, 400.33, 475.10, 549.86, 624.63]),
        labels=("1a", "2", "3", "4", "5"),
        note="PISA 2022 financial literacy cutpoints (NCES; L6 not available).",
    )
)

ALIASES.update(
    {
        "pisa2012_read": "pisa2012:reading",
        "pisa2015_read": "pisa2015:reading",
        "pisa2018_read": "pisa2018:reading",
        "pisa2022_read": "pisa2022:reading",
        "pisa2012_math": "pisa2012:math",
        "pisa2015_math": "pisa2015:math",
        "pisa2018_math": "pisa2018:math",
        "pisa2022_math": "pisa2022:math",
        "pisa2012_sci": "pisa2012:science",
        "pisa2015_sci": "pisa2015:science",
        "pisa2018_sci": "pisa2018:science",
        "pisa2022_sci": "pisa2022:science",
    }
)

register(
    Benchmark(
        key="naep:math:g4",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([214, 249, 282]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Mathematics Grade 4 achievement level cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="naep:math:g8",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([262, 299, 333]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Mathematics Grade 8 achievement level cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="naep:math:g12_post2005",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([141, 176, 216]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Mathematics Grade 12 cutpoints for 2005+ (NCES).",
    )
)

register(
    Benchmark(
        key="naep:reading:g4",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([208, 238, 268]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Reading Grade 4 cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="naep:reading:g8",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([243, 281, 323]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Reading Grade 8 cutpoints (NCES).",
    )
)
register(
    Benchmark(
        key="naep:reading:g12",
        kind="naep_achievement_levels",
        cutpoints=_as_tuple([265, 302, 346]),
        labels=("basic", "proficient", "advanced"),
        note="NAEP Reading Grade 12 cutpoints (NCES).",
    )
)

ALIASES.update(
    {
        "naep_math_g4": "naep:math:g4",
        "naep_math_g8": "naep:math:g8",
        "naep_math_g12": "naep:math:g12_post2005",
        "naep_read_g4": "naep:reading:g4",
        "naep_read_g8": "naep:reading:g8",
        "naep_read_g12": "naep:reading:g12",
    }
)
