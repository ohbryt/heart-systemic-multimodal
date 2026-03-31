"""
Reporting package for heart systemic multimodal project.

Provides:
- Markdown report generation with data manifest and figure catalog
- Publication-quality figure generation (UMAP, heatmap, Venn, Sankey, bar)
"""

from .report_generator import ReportGenerator
from .figure_generator import FigureGenerator

__all__ = ["ReportGenerator", "FigureGenerator"]
