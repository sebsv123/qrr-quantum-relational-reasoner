"""
QRR — Quantum Relational Reasoner
==================================
Late-collapse inference via coherent branch superposition.

Version: 0.2.0
Status:  Alpha — active research, API subject to change
Author:  Sebastián Sifontes Valentín
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Sebastián Sifontes Valentín"
__license__ = "MIT"

from qrr.qrr_model import QRRModel
from qrr.branch_bank import BranchBank
from qrr.collapse_index import CollapseIndex
from qrr.decoherence_module import DecoherenceModule

__all__ = [
    "QRRModel",
    "BranchBank",
    "CollapseIndex",
    "DecoherenceModule",
]
