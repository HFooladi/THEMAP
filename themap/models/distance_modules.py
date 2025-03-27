from Bio import Align
from Bio.Align import substitution_matrices
from Bio import SeqIO
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from themap.utils.logging import get_logger

logger = get_logger(__name__)


class ProteinIdentityDistance:
    def __init__(self, seq1: str, seq2: str) -> None:
        logger.debug(f"Initializing ProteinIdentityDistance with sequences of lengths {len(seq1)} and {len(seq2)}")
        self.seq1 = seq1
        self.seq2 = seq2
        self.aligner = Align.PairwiseAligner()
        self.aligner.open_gap_score = -11
        self.aligner.extend_gap_score = -1
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        logger.debug("Successfully initialized aligner with BLOSUM62 matrix")

    def get_seqid(self) -> float:
        return self.seqid

    def get_distance(self) -> float:
        logger.debug("Calculating protein identity distance")
        self.alignments = self.aligner.align(self.seq1, self.seq2)
        self.exact_match = self.alignments[0].counts().identities
        self.target_length = len(self.alignments[0].target)
        self.query_length = len(self.alignments[0].query)
        self.seqid = self.exact_match / min(self.query_length, self.target_length)
        self.distance = 1 - self.seqid
        logger.debug(f"Calculated distance: {self.distance:.4f} (sequence identity: {self.seqid:.4f})")
        return self.distance


def get_seqid(aligner: Align.PairwiseAligner, target: str, query: str) -> float:
    logger.debug(f"Calculating sequence identity between sequences of lengths {len(target)} and {len(query)}")
    alignments = aligner.align(target, query)
    exact_match = alignments[0].counts().identities
    target_length = len(alignments[0].target)
    query_length = len(alignments[0].query)
    seqid = exact_match / min(query_length, target_length)
    logger.debug(f"Calculated sequence identity: {seqid:.4f}")
    return seqid


def max_seqid(assay_sequence: Dict[str, Any], pdbbinds_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info(f"Finding maximum sequence identity for assay {assay_sequence.get('uniprot_id', 'unknown')} against {len(pdbbinds_list)} PDB structures")
    
    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    logger.debug("Initialized aligner with BLOSUM62 matrix")

    seqid = 0
    nearest_pdbbind = None
    
    for pdbbind in tqdm(pdbbinds_list, desc="Processing PDB structures"):
        try:
            new_seqid = get_seqid(aligner, pdbbind['seq'], assay_sequence['sequence'])
            if new_seqid > seqid:
                seqid = new_seqid
                nearest_pdbbind = pdbbind['id']
                logger.debug(f"Found new best match: {pdbbind['id']} with sequence identity {seqid:.4f}")
        except Exception as e:
            logger.warning(f"Error processing PDB structure {pdbbind.get('id', 'unknown')}: {str(e)}")
            # replace unknown aa in assay_sequence['sequence'] with 'X':
            modified_sequence = assay_sequence['sequence']
            for letter in assay_sequence['sequence']:
                if letter not in aligner.alphabet:
                    modified_sequence = modified_sequence.replace(letter, 'X')
                    logger.info(f"Replaced unknown amino acid {letter} with X in sequence {assay_sequence.get('uniprot_id', 'unknown')}")
            
            try:
                new_seqid = get_seqid(aligner, pdbbind['seq'], modified_sequence)
                if new_seqid > seqid:
                    seqid = new_seqid
                    nearest_pdbbind = pdbbind['id']
                    logger.debug(f"Found new best match after sequence modification: {pdbbind['id']} with sequence identity {seqid:.4f}")
            except Exception as e:
                logger.error(f"Failed to process PDB structure {pdbbind.get('id', 'unknown')} even after sequence modification: {str(e)}")

    assay_sequence['seqid'] = seqid
    assay_sequence['nearest_pdbbind'] = nearest_pdbbind
    logger.info(f"Completed processing assay {assay_sequence.get('uniprot_id', 'unknown')}. Best match: {nearest_pdbbind} with sequence identity {seqid:.4f}")
    return assay_sequence