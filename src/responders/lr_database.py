"""
Built-in curated ligand-receptor (LR) database.

Provides ~200 key LR pairs across major signaling categories:
growth_factors, cytokines, ECM, Wnt, Notch, TGFb, BMP.

Usage:
    db = LRDatabase()
    receptors = db.get_receptors("VEGFA")
    pairs = db.get_pairs_by_category("cytokines")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in curated LR pairs
# Each entry: (ligand, receptor, category, confidence, notes)
# ---------------------------------------------------------------------------
_BUILTIN_PAIRS: list[tuple[str, str, str, float, str]] = [
    # --- growth_factors ---
    ("VEGFA", "KDR", "growth_factors", 0.99, "angiogenesis"),
    ("VEGFA", "FLT1", "growth_factors", 0.98, "angiogenesis"),
    ("VEGFA", "NRP1", "growth_factors", 0.92, "co-receptor"),
    ("VEGFB", "FLT1", "growth_factors", 0.90, "angiogenesis"),
    ("VEGFC", "FLT4", "growth_factors", 0.95, "lymphangiogenesis"),
    ("FGF2", "FGFR1", "growth_factors", 0.97, "proliferation"),
    ("FGF2", "FGFR2", "growth_factors", 0.94, "proliferation"),
    ("FGF7", "FGFR2", "growth_factors", 0.92, "epithelial"),
    ("FGF21", "FGFR1", "growth_factors", 0.88, "metabolic"),
    ("FGF21", "KLB", "growth_factors", 0.95, "co-receptor/metabolic"),
    ("EGF", "EGFR", "growth_factors", 0.99, "proliferation"),
    ("HBEGF", "EGFR", "growth_factors", 0.95, "proliferation"),
    ("HBEGF", "ERBB4", "growth_factors", 0.88, "cardiac"),
    ("NRG1", "ERBB2", "growth_factors", 0.92, "cardiac"),
    ("NRG1", "ERBB3", "growth_factors", 0.90, "cardiac"),
    ("NRG1", "ERBB4", "growth_factors", 0.95, "cardiac/neural"),
    ("IGF1", "IGF1R", "growth_factors", 0.99, "growth"),
    ("IGF2", "IGF1R", "growth_factors", 0.96, "growth"),
    ("IGF2", "IGF2R", "growth_factors", 0.90, "growth"),
    ("PDGFA", "PDGFRA", "growth_factors", 0.97, "fibrosis"),
    ("PDGFB", "PDGFRB", "growth_factors", 0.98, "fibrosis/pericyte"),
    ("PDGFC", "PDGFRA", "growth_factors", 0.93, "fibrosis"),
    ("HGF", "MET", "growth_factors", 0.98, "regeneration"),
    ("MSTN", "ACVR2B", "growth_factors", 0.90, "muscle"),
    ("MSTN", "ACVR2A", "growth_factors", 0.88, "muscle"),
    ("GDF11", "ACVR2B", "growth_factors", 0.87, "aging/cardiac"),
    ("GDF15", "GFRAL", "growth_factors", 0.93, "cardiac stress"),
    ("EPO", "EPOR", "growth_factors", 0.97, "erythropoiesis"),
    ("THPO", "MPL", "growth_factors", 0.95, "megakaryocyte"),
    ("SCF", "KIT", "growth_factors", 0.96, "hematopoiesis"),
    # --- cytokines ---
    ("IL1A", "IL1R1", "cytokines", 0.97, "inflammation"),
    ("IL1B", "IL1R1", "cytokines", 0.99, "inflammation"),
    ("IL1B", "IL1RAP", "cytokines", 0.95, "co-receptor"),
    ("IL2", "IL2RA", "cytokines", 0.97, "T-cell"),
    ("IL4", "IL4R", "cytokines", 0.96, "Th2/fibrosis"),
    ("IL6", "IL6R", "cytokines", 0.99, "acute phase"),
    ("IL6", "IL6ST", "cytokines", 0.98, "signaling"),
    ("IL8", "CXCR1", "cytokines", 0.97, "neutrophil"),
    ("IL8", "CXCR2", "cytokines", 0.96, "neutrophil"),
    ("IL10", "IL10RA", "cytokines", 0.97, "anti-inflammatory"),
    ("IL10", "IL10RB", "cytokines", 0.95, "anti-inflammatory"),
    ("IL11", "IL11RA", "cytokines", 0.92, "cardiac fibrosis"),
    ("IL13", "IL13RA1", "cytokines", 0.94, "fibrosis"),
    ("IL15", "IL15RA", "cytokines", 0.93, "NK/T-cell"),
    ("IL17A", "IL17RA", "cytokines", 0.95, "inflammation"),
    ("IL18", "IL18R1", "cytokines", 0.93, "innate immunity"),
    ("IL21", "IL21R", "cytokines", 0.92, "B-cell"),
    ("IL33", "ST2", "cytokines", 0.96, "cardiac fibrosis"),
    ("IL33", "IL1RL1", "cytokines", 0.96, "alias ST2"),
    ("TSLP", "TSLPR", "cytokines", 0.88, "allergic"),
    ("TNF", "TNFRSF1A", "cytokines", 0.99, "inflammation/apoptosis"),
    ("TNF", "TNFRSF1B", "cytokines", 0.97, "inflammation"),
    ("TNFSF10", "TNFRSF10A", "cytokines", 0.94, "apoptosis"),
    ("TNFSF10", "TNFRSF10B", "cytokines", 0.93, "apoptosis"),
    ("IFNA1", "IFNAR1", "cytokines", 0.97, "antiviral"),
    ("IFNB1", "IFNAR1", "cytokines", 0.97, "antiviral"),
    ("IFNG", "IFNGR1", "cytokines", 0.99, "immune activation"),
    ("CXCL12", "CXCR4", "cytokines", 0.99, "homing/cardiac"),
    ("CXCL12", "CXCR7", "cytokines", 0.95, "scavenging"),
    ("CCL2", "CCR2", "cytokines", 0.98, "monocyte"),
    ("CCL5", "CCR5", "cytokines", 0.96, "T-cell"),
    ("CCL7", "CCR2", "cytokines", 0.92, "monocyte"),
    ("CCL11", "CCR3", "cytokines", 0.91, "eosinophil"),
    ("CX3CL1", "CX3CR1", "cytokines", 0.95, "microglia/monocyte"),
    ("HMGB1", "RAGE", "cytokines", 0.91, "DAMP"),
    ("HMGB1", "TLR4", "cytokines", 0.88, "DAMP"),
    ("ANGPT1", "TEK", "cytokines", 0.96, "vascular stability"),
    ("ANGPT2", "TEK", "cytokines", 0.92, "destabilization"),
    # --- ECM ---
    ("FN1", "ITGA5", "ECM", 0.97, "cell adhesion"),
    ("FN1", "ITGB1", "ECM", 0.98, "cell adhesion"),
    ("FN1", "ITGAV", "ECM", 0.94, "RGD binding"),
    ("LAMA1", "ITGA6", "ECM", 0.93, "basement membrane"),
    ("LAMA2", "ITGA7", "ECM", 0.94, "muscle"),
    ("LAMB1", "ITGA6", "ECM", 0.90, "basement membrane"),
    ("COL1A1", "DDR1", "ECM", 0.95, "collagen receptor"),
    ("COL1A1", "DDR2", "ECM", 0.93, "collagen receptor"),
    ("COL4A1", "ITGA1", "ECM", 0.91, "basement membrane"),
    ("COL6A1", "ITGA1", "ECM", 0.89, "fibrosis"),
    ("VTN", "ITGAV", "ECM", 0.94, "vitronectin"),
    ("VTN", "ITGB3", "ECM", 0.92, "vitronectin"),
    ("THBS1", "CD36", "ECM", 0.93, "antiangiogenic"),
    ("THBS1", "CD47", "ECM", 0.91, "don't-eat-me"),
    ("THBS2", "CD36", "ECM", 0.90, "antiangiogenic"),
    ("TNC", "ITGAV", "ECM", 0.91, "tenascin"),
    ("POSTN", "ITGAV", "ECM", 0.90, "cardiac fibrosis"),
    ("POSTN", "ITGB3", "ECM", 0.88, "cardiac fibrosis"),
    ("LTBP1", "TGFBR2", "ECM", 0.87, "TGFb sequestration"),
    ("SPARC", "ITGB1", "ECM", 0.86, "matrix modulation"),
    ("MMP2", "ITGAV", "ECM", 0.85, "proteolysis"),
    ("MMP9", "CD44", "ECM", 0.87, "ECM remodeling"),
    ("CTGF", "ITGAV", "ECM", 0.88, "fibrosis"),
    ("CTGF", "TrkA", "ECM", 0.82, "neural"),
    # --- Wnt ---
    ("WNT1", "FZD1", "Wnt", 0.90, "canonical"),
    ("WNT2", "FZD4", "Wnt", 0.88, "canonical"),
    ("WNT3A", "FZD1", "Wnt", 0.93, "canonical"),
    ("WNT3A", "LRP5", "Wnt", 0.92, "co-receptor"),
    ("WNT3A", "LRP6", "Wnt", 0.94, "co-receptor"),
    ("WNT4", "FZD6", "Wnt", 0.87, "canonical/non-canonical"),
    ("WNT5A", "FZD2", "Wnt", 0.91, "non-canonical"),
    ("WNT5A", "ROR2", "Wnt", 0.90, "non-canonical"),
    ("WNT5B", "FZD5", "Wnt", 0.86, "non-canonical"),
    ("WNT7A", "FZD9", "Wnt", 0.87, "neural"),
    ("WNT9B", "FZD4", "Wnt", 0.84, "kidney"),
    ("WNT10B", "FZD1", "Wnt", 0.89, "adipogenesis"),
    ("WNT16", "FZD5", "Wnt", 0.83, "bone"),
    ("RSPO1", "LGR4", "Wnt", 0.93, "Wnt potentiator"),
    ("RSPO2", "LGR5", "Wnt", 0.91, "Wnt potentiator"),
    ("RSPO3", "LGR4", "Wnt", 0.90, "Wnt potentiator"),
    ("DKK1", "LRP6", "Wnt", 0.92, "Wnt antagonist"),
    ("SFRP1", "FZD1", "Wnt", 0.88, "Wnt antagonist"),
    # --- Notch ---
    ("DLL1", "NOTCH1", "Notch", 0.96, "canonical Notch"),
    ("DLL3", "NOTCH1", "Notch", 0.91, "inhibitory"),
    ("DLL4", "NOTCH1", "Notch", 0.97, "angiogenesis"),
    ("DLL4", "NOTCH4", "Notch", 0.93, "endothelial"),
    ("JAG1", "NOTCH1", "Notch", 0.96, "canonical Notch"),
    ("JAG1", "NOTCH2", "Notch", 0.93, "canonical Notch"),
    ("JAG2", "NOTCH1", "Notch", 0.91, "canonical Notch"),
    ("JAG2", "NOTCH3", "Notch", 0.88, "vascular SMC"),
    ("NOTCH1", "NOTCH1", "Notch", 0.80, "cis-inhibition"),
    # --- TGFb ---
    ("TGFB1", "TGFBR1", "TGFb", 0.99, "canonical TGFb"),
    ("TGFB1", "TGFBR2", "TGFb", 0.99, "canonical TGFb"),
    ("TGFB2", "TGFBR1", "TGFb", 0.96, "canonical TGFb"),
    ("TGFB2", "TGFBR2", "TGFb", 0.96, "canonical TGFb"),
    ("TGFB3", "TGFBR1", "TGFb", 0.95, "canonical TGFb"),
    ("TGFB3", "TGFBR2", "TGFb", 0.94, "canonical TGFb"),
    ("TGFB1", "TGFBR3", "TGFb", 0.90, "co-receptor"),
    ("TGFB2", "TGFBR3", "TGFb", 0.92, "co-receptor"),
    ("TGFB1", "ENG", "TGFb", 0.91, "endoglin/endothelial"),
    ("ACTIVIN_A", "ACVR1B", "TGFb", 0.94, "activin signaling"),
    ("ACTIVIN_A", "ACVR2A", "TGFb", 0.93, "activin signaling"),
    ("ACTIVIN_B", "ACVR2A", "TGFb", 0.90, "activin signaling"),
    ("INHIBIN_A", "ACVR2A", "TGFb", 0.85, "inhibitory"),
    ("NODAL", "ACVR1B", "TGFb", 0.88, "embryonic"),
    ("LEFTY1", "ACVR2A", "TGFb", 0.83, "antagonist"),
    ("LEFTY2", "ACVR2B", "TGFb", 0.82, "antagonist"),
    # --- BMP ---
    ("BMP2", "BMPR1A", "BMP", 0.97, "osteogenesis/cardiac"),
    ("BMP2", "BMPR2", "BMP", 0.96, "osteogenesis"),
    ("BMP4", "BMPR1A", "BMP", 0.96, "cardiac/vascular"),
    ("BMP4", "BMPR2", "BMP", 0.95, "cardiac/vascular"),
    ("BMP5", "BMPR1B", "BMP", 0.90, "cartilage"),
    ("BMP6", "BMPR1A", "BMP", 0.91, "iron homeostasis"),
    ("BMP6", "HJV", "BMP", 0.89, "hepcidin/iron"),
    ("BMP7", "BMPR1A", "BMP", 0.95, "kidney/cardiac"),
    ("BMP7", "BMPR2", "BMP", 0.94, "anti-fibrosis"),
    ("BMP9", "ACVRL1", "BMP", 0.95, "pulmonary vascular"),
    ("BMP9", "BMPR2", "BMP", 0.93, "PAH"),
    ("BMP10", "ACVRL1", "BMP", 0.92, "cardiac"),
    ("BMP10", "BMPR2", "BMP", 0.90, "cardiac"),
    ("GDF5", "BMPR1B", "BMP", 0.91, "joint/skeletal"),
    ("NOGGIN", "BMPR1A", "BMP", 0.93, "BMP antagonist"),
    ("CHORDIN", "BMPR1A", "BMP", 0.88, "BMP antagonist"),
    ("GREMLIN1", "BMPR2", "BMP", 0.87, "BMP antagonist/fibrosis"),
    ("GREMLIN2", "BMPR1A", "BMP", 0.83, "BMP antagonist"),
    # --- additional cardiac/metabolic ---
    ("ANP", "NPRA", "growth_factors", 0.97, "cardiac/natriuretic"),
    ("BNP", "NPRA", "growth_factors", 0.96, "cardiac/natriuretic"),
    ("CNP", "NPRB", "growth_factors", 0.95, "vascular"),
    ("ADM", "CALCRL", "growth_factors", 0.92, "vasodilation"),
    ("APLN", "APLNR", "growth_factors", 0.93, "cardiac contractility"),
    ("EDN1", "EDNRA", "cytokines", 0.97, "vasoconstriction"),
    ("EDN1", "EDNRB", "cytokines", 0.93, "endothelial"),
    ("EDN2", "EDNRA", "cytokines", 0.91, "vasoconstriction"),
    ("ANGII", "AGTR1", "cytokines", 0.97, "RAAS/cardiac"),
    ("ANGII", "AGTR2", "cytokines", 0.92, "cardioprotective"),
    ("LEP", "LEPR", "cytokines", 0.97, "metabolic/cardiac"),
    ("ADIPOQ", "ADIPOR1", "cytokines", 0.95, "metabolic"),
    ("ADIPOQ", "ADIPOR2", "cytokines", 0.93, "hepatic"),
    ("RETN", "CAP1", "cytokines", 0.85, "insulin resistance"),
    ("FGF19", "FGFR4", "growth_factors", 0.93, "bile acid/metabolic"),
    ("FGF23", "FGFR1", "growth_factors", 0.90, "phosphate/cardiac"),
    ("FGF23", "KLB", "growth_factors", 0.92, "co-receptor"),
    ("SEMA3A", "NRP1", "ECM", 0.91, "guidance/cardiac"),
    ("SEMA3C", "PLXNA2", "ECM", 0.87, "cardiac outflow"),
    ("EFNA1", "EPHA2", "ECM", 0.90, "ephrin/contact"),
    ("EFNB1", "EPHB2", "ECM", 0.89, "ephrin/contact"),
    ("PROS1", "TYRO3", "ECM", 0.88, "anticoagulation"),
    ("GAS6", "AXL", "ECM", 0.91, "efferocytosis/fibrosis"),
    ("GAS6", "MERTK", "ECM", 0.90, "efferocytosis"),
    ("PROS1", "MERTK", "ECM", 0.87, "efferocytosis"),
    # --- Hedgehog ---
    ("SHH", "PTCH1", "growth_factors", 0.97, "Hedgehog pathway"),
    ("IHH", "PTCH1", "growth_factors", 0.94, "Hedgehog pathway"),
    ("DHH", "PTCH1", "growth_factors", 0.90, "Hedgehog pathway"),
    # --- Complement / coagulation ---
    ("C3", "CR1", "cytokines", 0.89, "complement"),
    ("C3", "CR3", "cytokines", 0.87, "complement"),
    ("C5A", "C5AR1", "cytokines", 0.93, "complement/inflammation"),
    ("THROMBIN", "PAR1", "cytokines", 0.95, "coagulation/signaling"),
    ("THROMBIN", "PAR4", "cytokines", 0.90, "coagulation/platelet"),
]


@dataclass
class LRPair:
    """Represents a single ligand-receptor pair with metadata."""

    ligand: str
    receptor: str
    category: str
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "ligand": self.ligand,
            "receptor": self.receptor,
            "category": self.category,
            "confidence": self.confidence,
            "notes": self.notes,
        }


class LRDatabase:
    """
    Curated ligand-receptor database for cross-organ signaling analysis.

    Loads from built-in pairs by default. Optionally augments from a YAML
    or CSV file.

    Parameters
    ----------
    extra_path : Path, optional
        Path to a YAML (list of dicts) or CSV file with additional LR pairs.
        Expected YAML schema::

            - ligand: VEGFA
              receptor: KDR
              category: growth_factors
              confidence: 0.99
              notes: angiogenesis

    Examples
    --------
    >>> db = LRDatabase()
    >>> db.get_receptors("TGFB1")
    ['TGFBR1', 'TGFBR2', 'TGFBR3', 'ENG']
    """

    def __init__(self, extra_path: Optional[Path] = None) -> None:
        self._pairs: list[LRPair] = self._load_builtin()
        if extra_path is not None:
            self._pairs.extend(self._load_extra(extra_path))
        self._df: pd.DataFrame = self._to_dataframe()
        logger.info(
            "LRDatabase initialised with %d pairs across %d categories",
            len(self._pairs),
            self._df["category"].nunique(),
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_builtin() -> list[LRPair]:
        return [
            LRPair(ligand=lig, receptor=rec, category=cat, confidence=conf, notes=notes)
            for lig, rec, cat, conf, notes in _BUILTIN_PAIRS
        ]

    @staticmethod
    def _load_extra(path: Path) -> list[LRPair]:
        """Load extra LR pairs from YAML or CSV."""
        path = Path(path)
        pairs: list[LRPair] = []
        if not path.exists():
            logger.warning("Extra LR database not found: %s", path)
            return pairs
        try:
            if path.suffix in {".yaml", ".yml"}:
                with path.open() as fh:
                    records = yaml.safe_load(fh) or []
                for r in records:
                    pairs.append(
                        LRPair(
                            ligand=r["ligand"],
                            receptor=r["receptor"],
                            category=r.get("category", "custom"),
                            confidence=float(r.get("confidence", 0.8)),
                            notes=r.get("notes", ""),
                        )
                    )
            elif path.suffix == ".csv":
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    pairs.append(
                        LRPair(
                            ligand=str(row["ligand"]),
                            receptor=str(row["receptor"]),
                            category=str(row.get("category", "custom")),
                            confidence=float(row.get("confidence", 0.8)),
                            notes=str(row.get("notes", "")),
                        )
                    )
            else:
                logger.warning("Unsupported file format for extra LR pairs: %s", path.suffix)
        except Exception as exc:
            logger.error("Failed to load extra LR pairs from %s: %s", path, exc)
        logger.info("Loaded %d extra LR pairs from %s", len(pairs), path)
        return pairs

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self._pairs])

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_receptors(self, ligand: str) -> list[str]:
        """
        Return all known receptors for the given ligand.

        Parameters
        ----------
        ligand : str
            Gene symbol of the ligand (case-insensitive).

        Returns
        -------
        list[str]
            Sorted list of receptor gene symbols.
        """
        mask = self._df["ligand"].str.upper() == ligand.upper()
        return sorted(self._df.loc[mask, "receptor"].tolist())

    def get_ligands(self, receptor: str) -> list[str]:
        """
        Return all known ligands for the given receptor.

        Parameters
        ----------
        receptor : str
            Gene symbol of the receptor (case-insensitive).

        Returns
        -------
        list[str]
            Sorted list of ligand gene symbols.
        """
        mask = self._df["receptor"].str.upper() == receptor.upper()
        return sorted(self._df.loc[mask, "ligand"].tolist())

    def get_pairs_by_category(self, category: str) -> pd.DataFrame:
        """
        Return all LR pairs belonging to a specific category.

        Parameters
        ----------
        category : str
            One of: growth_factors, cytokines, ECM, Wnt, Notch, TGFb, BMP.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe of LR pairs.
        """
        mask = self._df["category"].str.lower() == category.lower()
        return self._df.loc[mask].copy().reset_index(drop=True)

    def get_pair_confidence(self, ligand: str, receptor: str) -> float:
        """
        Return the confidence score for a specific LR pair.

        Returns 0.0 if the pair is not found.
        """
        mask = (self._df["ligand"].str.upper() == ligand.upper()) & (
            self._df["receptor"].str.upper() == receptor.upper()
        )
        rows = self._df.loc[mask, "confidence"]
        if rows.empty:
            return 0.0
        return float(rows.iloc[0])

    def pair_exists(self, ligand: str, receptor: str) -> bool:
        """Check whether a given LR pair exists in the database."""
        return self.get_pair_confidence(ligand, receptor) > 0.0

    @property
    def categories(self) -> list[str]:
        """List of unique signaling categories present in the database."""
        return sorted(self._df["category"].unique().tolist())

    @property
    def all_ligands(self) -> list[str]:
        """Sorted list of all unique ligands in the database."""
        return sorted(self._df["ligand"].unique().tolist())

    @property
    def all_receptors(self) -> list[str]:
        """Sorted list of all unique receptors in the database."""
        return sorted(self._df["receptor"].unique().tolist())

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full database as a pandas DataFrame (copy)."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._pairs)

    def __repr__(self) -> str:
        return f"LRDatabase(pairs={len(self)}, categories={self.categories})"
