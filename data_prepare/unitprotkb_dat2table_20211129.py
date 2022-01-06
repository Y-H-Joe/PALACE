#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 20:13:23 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================
ID   A0A133N9B4_CLOPF        Unreviewed;       312 AA.
AC   A0A133N9B4;
DT   08-JUN-2016, integrated into UniProtKB/TrEMBL.
DT   08-JUN-2016, sequence version 1.
DT   02-JUN-2021, entry version 32.
DE   RecName: Full=Thioredoxin reductase {ECO:0000256|RuleBase:RU003880};
DE            EC=1.8.1.9 {ECO:0000256|RuleBase:RU003880};
GN   Name=trxB {ECO:0000313|EMBL:NGT57282.1};
GN   Synonyms=trxB_2 {ECO:0000313|EMBL:SQB60347.1}, trxB_3
GN   {ECO:0000313|EMBL:SUY42265.1};
GN   ORFNames=G6Z03_12015 {ECO:0000313|EMBL:NGT85062.1}, G6Z15_05390
GN   {ECO:0000313|EMBL:NGT57282.1}, G6Z33_05235
GN   {ECO:0000313|EMBL:NGU41308.1}, G6Z34_11915
GN   {ECO:0000313|EMBL:NGU30808.1}, G6Z38_11860
GN   {ECO:0000313|EMBL:NGU10459.1}, HMPREF3222_01054
GN   {ECO:0000313|EMBL:KXA12890.1}, JFP838_14660
GN   {ECO:0000313|EMBL:AMN36930.1}, NCTC10240_02512
GN   {ECO:0000313|EMBL:SUY31881.1}, NCTC10578_02785
GN   {ECO:0000313|EMBL:SUY42265.1}, NCTC10719_01946
GN   {ECO:0000313|EMBL:SQB60347.1}, NCTC3182_01326
GN   {ECO:0000313|EMBL:STB44382.1}, NCTC8503_01914
GN   {ECO:0000313|EMBL:VTQ60426.1};
OS   Clostridium perfringens.
OC   Bacteria; Firmicutes; Clostridia; Eubacteriales; Clostridiaceae;
OC   Clostridium.
OX   NCBI_TaxID=1502 {ECO:0000313|EMBL:KXA12890.1, ECO:0000313|Proteomes:UP000070646};
RN   [1] {ECO:0000313|EMBL:AMN36930.1, ECO:0000313|Proteomes:UP000070260}
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RC   STRAIN=JP838 {ECO:0000313|EMBL:AMN36930.1,
RC   ECO:0000313|Proteomes:UP000070260};
RX   PubMed=26859667;
RA   Mehdizadeh Gohari I., Kropinski A.M., Weese S.J., Parreira V.R.,
RA   Whitehead A.E., Boerlin P., Prescott J.F.;
RT   "Plasmid Characterization and Chromosome Analysis of Two netF+ Clostridium
RT   perfringens Isolates Associated with Foal and Canine Necrotizing
RT   Enteritis.";
RL   PLoS ONE 11:E0148344-E0148344(2016).
RN   [2] {ECO:0000313|EMBL:KXA12890.1, ECO:0000313|Proteomes:UP000070646}
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RC   STRAIN=MJR7757A {ECO:0000313|EMBL:KXA12890.1,
RC   ECO:0000313|Proteomes:UP000070646};
RA   Oliw E.H.;
RL   Submitted (JAN-2016) to the EMBL/GenBank/DDBJ databases.
RN   [3] {ECO:0000313|Proteomes:UP000249986, ECO:0000313|Proteomes:UP000253871}
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RC   STRAIN=NCTC10240 {ECO:0000313|EMBL:SUY31881.1,
RC   ECO:0000313|Proteomes:UP000255324}, NCTC10578
RC   {ECO:0000313|EMBL:SUY42265.1, ECO:0000313|Proteomes:UP000254715},
RC   NCTC10719 {ECO:0000313|EMBL:SQB60347.1,
RC   ECO:0000313|Proteomes:UP000249986}, and NCTC3182
RC   {ECO:0000313|EMBL:STB44382.1, ECO:0000313|Proteomes:UP000253871};
RG   Pathogen Informatics;
RA   Doyle S.;
RL   Submitted (JUN-2018) to the EMBL/GenBank/DDBJ databases.
RN   [4] {ECO:0000313|EMBL:VTQ60426.1, ECO:0000313|Proteomes:UP000396299}
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RC   STRAIN=NCTC8503 {ECO:0000313|EMBL:VTQ60426.1,
RC   ECO:0000313|Proteomes:UP000396299};
RG   Pathogen Informatics;
RL   Submitted (MAY-2019) to the EMBL/GenBank/DDBJ databases.
RN   [5] {ECO:0000313|Proteomes:UP000475880, ECO:0000313|Proteomes:UP000480682}
RP   NUCLEOTIDE SEQUENCE [LARGE SCALE GENOMIC DNA].
RC   STRAIN=CP-09 {ECO:0000313|EMBL:NGT85062.1,
RC   ECO:0000313|Proteomes:UP000481575}, CP-21
RC   {ECO:0000313|EMBL:NGT57282.1, ECO:0000313|Proteomes:UP000480682},
RC   CP-39 {ECO:0000313|EMBL:NGU41308.1,
RC   ECO:0000313|Proteomes:UP000475880}, CP-40
RC   {ECO:0000313|EMBL:NGU30808.1, ECO:0000313|Proteomes:UP000481454}, and
RC   CP-44 {ECO:0000313|EMBL:NGU10459.1,
RC   ECO:0000313|Proteomes:UP000491003};
RA   Feng Y., Hu Y.;
RT   "Genomic Insights into the Phylogeny and Genetic Plasticity of the Human
RT   and Animal Enteric Pathogen Clostridium perfringens.";
RL   Submitted (FEB-2020) to the EMBL/GenBank/DDBJ databases.
CC   -!- CATALYTIC ACTIVITY:
CC       Reaction=[thioredoxin]-dithiol + NADP(+) = [thioredoxin]-disulfide +
CC         H(+) + NADPH; Xref=Rhea:RHEA:20345, Rhea:RHEA-COMP:10698, Rhea:RHEA-
CC         COMP:10700, ChEBI:CHEBI:15378, ChEBI:CHEBI:29950, ChEBI:CHEBI:50058,
CC         ChEBI:CHEBI:57783, ChEBI:CHEBI:58349; EC=1.8.1.9;
CC         Evidence={ECO:0000256|RuleBase:RU003880};
CC   -!- COFACTOR:
CC       Name=FAD; Xref=ChEBI:CHEBI:57692;
CC         Evidence={ECO:0000256|RuleBase:RU003880};
CC   -!- COFACTOR:
CC       Name=FAD; Xref=ChEBI:CHEBI:57692;
CC         Evidence={ECO:0000256|RuleBase:RU003881};
CC       Note=Binds 1 FAD per subunit. {ECO:0000256|RuleBase:RU003881};
CC   -!- SUBUNIT: Homodimer. {ECO:0000256|RuleBase:RU003880}.
CC   -!- SIMILARITY: Belongs to the class-II pyridine nucleotide-disulfide
CC       oxidoreductase family. {ECO:0000256|ARBA:ARBA00009333,
CC       ECO:0000256|RuleBase:RU003880}.
CC   ---------------------------------------------------------------------------
CC   Copyrighted by the UniProt Consortium, see https://www.uniprot.org/terms
CC   Distributed under the Creative Commons Attribution (CC BY 4.0) License
CC   ---------------------------------------------------------------------------
DR   EMBL; CP010994; AMN36930.1; -; Genomic_DNA.
DR   EMBL; LRPU01000054; KXA12890.1; -; Genomic_DNA.
DR   EMBL; JAALMS010000003; NGT57282.1; -; Genomic_DNA.
DR   EMBL; JAALNE010000006; NGT85062.1; -; Genomic_DNA.
DR   EMBL; JAALLV010000003; NGU10459.1; -; Genomic_DNA.
DR   EMBL; JAALLZ010000004; NGU30808.1; -; Genomic_DNA.
DR   EMBL; JAALMA010000004; NGU41308.1; -; Genomic_DNA.
DR   EMBL; UAWG01000012; SQB60347.1; -; Genomic_DNA.
DR   EMBL; UFWQ01000001; STB44382.1; -; Genomic_DNA.
DR   EMBL; UFXH01000002; SUY31881.1; -; Genomic_DNA.
DR   EMBL; UFXA01000002; SUY42265.1; -; Genomic_DNA.
DR   EMBL; CABEEO010000008; VTQ60426.1; -; Genomic_DNA.
DR   RefSeq; WP_003452248.1; NZ_UWOU01000021.1.
DR   EnsemblBacteria; AMN36930; AMN36930; JFP838_14660.
DR   EnsemblBacteria; KXA12890; KXA12890; HMPREF3222_01054.
DR   EnsemblBacteria; SQB60347; SQB60347; NCTC10719_01946.
DR   EnsemblBacteria; STB44382; STB44382; NCTC3182_01326.
DR   EnsemblBacteria; SUY31881; SUY31881; NCTC10240_02512.
DR   EnsemblBacteria; SUY42265; SUY42265; NCTC10578_02785.
DR   GeneID; 29570142; -.
DR   PATRIC; fig|1502.174.peg.1070; -.
DR   Proteomes; UP000070260; Chromosome.
DR   Proteomes; UP000070646; Unassembled WGS sequence.
DR   Proteomes; UP000249986; Unassembled WGS sequence.
DR   Proteomes; UP000253871; Unassembled WGS sequence.
DR   Proteomes; UP000254715; Unassembled WGS sequence.
DR   Proteomes; UP000255324; Unassembled WGS sequence.
DR   Proteomes; UP000396299; Unassembled WGS sequence.
DR   Proteomes; UP000475880; Unassembled WGS sequence.
DR   Proteomes; UP000480682; Unassembled WGS sequence.
DR   Proteomes; UP000481454; Unassembled WGS sequence.
DR   Proteomes; UP000481575; Unassembled WGS sequence.
DR   Proteomes; UP000491003; Unassembled WGS sequence.
DR   GO; GO:0005737; C:cytoplasm; IEA:InterPro.
DR   GO; GO:0004791; F:thioredoxin-disulfide reductase activity; IEA:UniProtKB-UniRule.
DR   GO; GO:0019430; P:removal of superoxide radicals; IEA:UniProtKB-UniRule.
DR   Gene3D; 3.50.50.60; -; 2.
DR   InterPro; IPR036188; FAD/NAD-bd_sf.
DR   InterPro; IPR023753; FAD/NAD-binding_dom.
DR   InterPro; IPR008255; Pyr_nucl-diS_OxRdtase_2_AS.
DR   InterPro; IPR005982; Thioredox_Rdtase.
DR   Pfam; PF07992; Pyr_redox_2; 1.
DR   SUPFAM; SSF51905; SSF51905; 1.
DR   TIGRFAMs; TIGR01292; TRX_reduct; 1.
DR   PROSITE; PS00573; PYRIDINE_REDOX_2; 1.
PE   3: Inferred from homology;
KW   Disulfide bond {ECO:0000256|ARBA:ARBA00023157};
KW   FAD {ECO:0000256|RuleBase:RU003880};
KW   Flavoprotein {ECO:0000256|RuleBase:RU003880};
KW   NADP {ECO:0000256|RuleBase:RU003881};
KW   Oxidoreductase {ECO:0000256|RuleBase:RU003880,
KW   ECO:0000313|EMBL:NGT57282.1};
KW   Redox-active center {ECO:0000256|ARBA:ARBA00023284,
KW   ECO:0000256|RuleBase:RU003880}.
FT   DOMAIN          9..295
FT                   /note="Pyr_redox_2"
FT                   /evidence="ECO:0000259|Pfam:PF07992"
SQ   SEQUENCE   312 AA;  34312 MW;  80AFF91BCB3CAA72 CRC64;
     MDKEIKELDL IIIGAGPAGL TSAIYASRAK LSTLVLEDNL VGGQVRSTYT VENYPGFTEI
     TGNDLADRIQ AQAEACGAII DEFDFIENVS LKDDEKIIET GDYIYKPKAV IIATGATPKK
     LPIPSESKYL GKGVHYCAVC DGAVYQDEVV AVVGGGNAAL EEALYLSNIV KKVIVIRRYD
     YFRAEAKTLE AASNKENIEI MYNWDLVDVL GGEFVEAARI KNTKTGEEKE IAINGVFGYI
     GTEPKTSMFR EYINVKENGY IEGDENMRTN VKGVYVAGDV REKMFRQITT AVSDGTIAAL
     HAEKYISEIK ER
//
=================================== output ====================================
UnitProtID	taxaID		RheaID		EC	reaction	CHEBI	protein
================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import os
from itertools import islice
import re
import sys

dp=sys.argv[1]
#dp='aa'
output1=str(dp+".enzyme.tsv")
output2=str(dp+".nonenzyme.tsv")
#accession=os.path.basename(dp)

def error_log(instance,index,value,exit=True):
    print(instance," error , index: ",index)
    print(instance," error , value: ",value)
    if exit:
        sys.exit()

enzyme_record_list=[]
nonenzyme_record_list=[]
with open(dp) as f:

    lines=f.readlines()

    UnitProtID=''
    RheaID=''
    taxaID=''
    EC=''
    reaction=''
    protein=''
    CHEBI=''
    
    chunk=False
    catalytic=0
    RheaID_EC_reaction_CHEBI_list=[]
    
    for index,value in enumerate(lines):
        ## UnitProtID
        if value.startswith('AC   '):
            UnitProtID_line=lines[index]
            UnitProtID=UnitProtID_line.split()[1].strip(';').strip()

            next(islice(lines,1,1), None)
        ## EC
        ## reaction
        ## RheaID
        ## CHEBI
        ## one enzyme might have multiple reaction or no reaction
        if value.startswith('CC       Reaction='):
            catalytic+=1
            
            cc_line_num=1
            activity=value.replace('CC       Reaction=','').strip()
            try:
                while lines[index+cc_line_num].startswith('CC       '):
                    temp=lines[index+cc_line_num].replace('CC   ','').strip()
                    activity+=temp
                    cc_line_num+=1
            except:
                print('file truncated.')
            
            value_split=[x for x in activity.split(';') if x!='' and x!='\n']
            ## reaction
            reaction=value_split[0]
            ## RheaID
            try:
                RheaID=re.findall('Xref=Rhea:RHEA:\d+,', activity)[0].replace('Xref=Rhea:RHEA:','').strip(',')
            except:
                RheaID='-'
                error_log('RheaID',index,activity,exit=False)
            ## EC
            try:
                EC=re.findall('EC=.+\..+\..+\..+?;', activity)[0].replace('EC=','').strip(';').strip()
            except:
                EC='-.-.-.-'
                error_log('EC',index,activity,exit=False)
            ## CHEBI
            try:
                CHEBI=';'.join([x.replace('CHEBI:','') for x in re.findall('CHEBI:\d+', activity)])
            except:
                CHEBI='-'
                error_log('CHEBI',index,activity,exit=False)
                
            RheaID_EC_reaction_CHEBI_list.append([RheaID,EC,reaction,CHEBI])
                
            next(islice(lines,index,index+cc_line_num), None)
        ## taxaID
        if value.startswith('OX   '):
            try:
                taxaID=re.findall('NCBI_TaxID=\d+',value)[0].replace('NCBI_TaxID=','').strip()
            except:
                error_log('taxaID',index,value,exit=False)
                sys.exit()
        ## protein
        if value.startswith('SQ   '):
            protein_line_num=1
            try:
                while lines[index+protein_line_num].startswith('     '):
                    protein=protein+''.join([x.strip() for x in re.split('"|=| ',lines[index+protein_line_num]) if x!='' and x!='\n'])
                    protein_line_num+=1
            except:
                protein='-'
            next(islice(lines,index,index+protein_line_num), None)
        ## new block
        if value.startswith('//'):
            ## enzyme
            if catalytic > 0:
                #try:
                for RErC in RheaID_EC_reaction_CHEBI_list:
                        
                    enzyme='\t'.join([UnitProtID,taxaID,'\t'.join(map(str,RErC)),protein])
                    enzyme_record_list.append(enzyme)
                #except:
                #    pass
            ## nonenzyme
            if catalytic == 0:
                nonenzyme='\t'.join([UnitProtID,taxaID,protein])
                nonenzyme_record_list.append(nonenzyme)
                
            ##new start
            chunk=False
            UnitProtID=''
            RheaID=''
            taxaID=''
            EC=''
            reaction=''
            protein=''
            CHEBI=''
            catalytic=0
            RheaID_EC_reaction_CHEBI_list=[]

            

with open(output1,'w') as w:
    #w.write('accession\tgeneID\tUnitProtID\tNCBI\tCOG\tPfam\tInterPro\tprotein\n')
    w.write('UnitProtID\ttaxaID\tRheaID\tEC\treaction\tCHEBI\tprotein\n')
    for enzyme in enzyme_record_list:
        w.write(enzyme)
        w.write('\n')
        
with open(output2,'w') as w:
    #w.write('accession\tgeneID\tUnitProtID\tNCBI\tCOG\tPfam\tInterPro\tprotein\n')
    w.write('UnitProtID\ttaxaID\tprotein\n')
    for nonenzyme in nonenzyme_record_list:
        w.write(nonenzyme)
        w.write('\n')


