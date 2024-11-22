from Bio import Align
from Bio.Align import substitution_matrices
from Bio import SeqIO


class ProteinIdentityDistance():
    def __init__(self, seq1, seq2):
        self.seq1 = seq1
        self.seq2 = seq2
        self.aligner = Align.PairwiseAligner()
        self.aligner.open_gap_score = -11
        self.aligner.extend_gap_score = -1
        self.aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")


    def get_seqid(self):
        return self.seqid

    def get_distance(self):
        self.alignments = self.aligner.align(seq1, seq2)
        self.exact_match = self.alignments[0].counts().identities
        self.target_length = len(self.alignments[0].target)
        self.query_length = len(self.alignments[0].query)
        self.seqid = self.exact_match / min(self.query_length, self.target_length)
        self.distance = 1 - self.seqid
        return self.distance



def get_seqid(aligner, target, query):
    alignments = aligner.align(target, query)
    exact_match = alignments[0].counts().identities
    target_length = len(alignments[0].target)
    query_length = len(alignments[0].query)
    seqid = exact_match / min(query_length, target_length)
    return seqid


def max_seqid(assay_sequence, pdbbinds_list):

    aligner = Align.PairwiseAligner()
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

    seqid = 0
    for pdbbind in tqdm(pdbbinds_list):
        try:
            new_seqid = get_seqid(aligner, pdbbind['seq'], assay_sequence['sequence'])
        except:
            # replace unknown aa in assay_sequence['sequence'] with 'X':
            for letter in assay_sequence['sequence']:
                if letter not in aligner.alphabet:
                    assay_sequence['sequence'] = assay_sequence['sequence'].replace(letter, 'X')
                    #logger.info(f"Unknown amino acid {letter} in {assay_sequence['uniprot_id']}. Replaced with X")
                    new_seqid = get_seqid(aligner, pdbbind['seq'], assay_sequence['sequence'])
                    
        if new_seqid > seqid:
            seqid = new_seqid
            nearest_pdbbind = pdbbind['id']

    assay_sequence['seqid'] = seqid
    assay_sequence['nearest_pdbbind'] = nearest_pdbbind
    return assay_sequence