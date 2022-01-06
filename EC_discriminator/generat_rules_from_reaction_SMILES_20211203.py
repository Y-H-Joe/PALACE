#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:04:20 2021

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ discription ==============================####

#================================== input =====================================

#================================== output ====================================

#================================ parameters ==================================

#================================== example ===================================

#================================== warning ===================================

####=======================================================================####
"""
from rdkit.Chem import rdFMCS
from rdkit import Chem
mol1 = Chem.MolFromSmiles("O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C")
mol2 = Chem.MolFromSmiles("CC(C)CCCCCC(=O)NCC1=CC(=C(C=C1)O)OC")
mol3 = Chem.MolFromSmiles("c1(C=O)cc(OC)c(O)cc1")
mols = [mol1,mol2,mol3]
res=rdFMCS.FindMCS(mols)
common=Chem.MolFromSmarts(res.smartsString)

mols = [Chem.MolFromSmiles("Nc1ccccc1"*10), Chem.MolFromSmiles("Nc1ccccccccc1"*10)]
rs=rdFMCS.FindMCS(mols, timeout=1)
print(rs.canceled)
print(rs.smartsString)



#############
smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles(smi)
print(type(mol))
from rdkit import Chem
mol = Chem.MolFromMolFile('rd.mol')
print(type(mol))
seq='GGGGG'
mol = Chem.MolFromSequence(seq)
smi = Chem.MolToSmiles(mol)
print("smi",smi)

smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
smi = Chem.MolToSmiles(mol)
print(smi)
molblock = Chem.MolToMolBlock(mol)
print(molblock)
print(molblock,file=open('foo.mol','w+'))

from rdkit import Chem
smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
atoms = mol.GetAtoms()
print(type(atoms))
print(type(atoms[0]))

from rdkit import Chem
smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
bonds = mol.GetBonds()
print(type(bonds))
print(type(bonds[0]))

from rdkit import Chem
smi='CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3'
mol = Chem.MolFromSmiles(smi)
atom0 = mol.GetAtomWithIdx(0)
print(type(atom0))

mol.GetConformer().GetAtomPosition(1)[0]
mol.GetConformer().GetAtomPosition(1).x
mol.GetConformer().GetAtomPosition(1).y
mol.GetConformer().GetAtomPosition(1).z
x,y,z =mol.GetConformer().GetAtomPosition(1)
xyz = list(mol.GetConformer().GetAtomPosition(3))

from rdkit import Chem
m = Chem.MolFromSmiles('OC1C2C1CC2')
atom2  = m.GetAtomWithIdx(2)
print("atom2 in ring:",atom2.IsInRing())
print("atom2 in 3-ring:",atom2.IsInRingSize(3))
print("atom2 in 4-ring:",atom2.IsInRingSize(4))
print("atom2 in 5-ring:",atom2.IsInRingSize(5))

from rdkit import Chem
m = Chem.MolFromSmiles('OC1C2C1CC2')
ssr = Chem.GetSymmSSSR(m)
num_ring = len(ssr)
print("num of ring",num_ring)
for ring in ssr:
    print("ring consisted of atoms id:",list(ring))
    
    
from rdkit import Chem
m = Chem.MolFromSmiles('OCCC')
inchi = Chem.inchi.MolToInchi(m)
print(inchi)

from rdkit import Chem
m = Chem.MolFromSmiles('OC1C2C1CC2')
ri = m.GetRingInfo()
print(type(ri))

from rdkit import Chem
m = Chem.MolFromSmiles('OC1C2C1CC2')
m2 = Chem.AddHs(m)
print("m Smiles:",Chem.MolToSmiles(m))
print("m2 Smiles:",Chem.MolToSmiles(m2))
print("num ATOMs in m:",m2.GetNumAtoms())
print("num ATOMs in m2:",m.GetNumAtoms())

m = Chem.MolFromSmiles('c1ccccc1')
for bond in m.GetBonds():
    print(bond.GetBondType())

m = Chem.MolFromSmiles('c1ccccc1')
Chem.Kekulize(m)
for bond in m.GetBonds():
    print(bond.GetBondType())

print("bond 1 is aromatic",m.GetBondWithIdx(1).GetIsAromatic())
print("atom 1 is aromatic",m.GetAtomWithIdx(1).GetIsAromatic())

from rdkit.Chem import Draw
from rdkit import Chem
smis=[
    'COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3',
#     'CCN(CC1=C(C=CC(=C1)C(F)(F)F)C2=CC(=C3N2C=NC=C3)CC(=O)O)C(=O)C4CC4',
    'C1=CC2=C(C(=C1)C3=CN=CN4C3=CC=C4)ON=C2C5=CC=C(C=C5)F',
    'COC(=O)C1=CC2=CC=CN2C=N1',
    'C1=C2C=C(N=CN2C(=C1)Cl)C(=O)O',
]
template = Chem.MolFromSmiles('c1nccc2n1ccc2')
AllChem.Compute2DCoords(template)
mols=[]
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    AllChem.GenerateDepictionMatching2DStructure(mol,template)
    mols.append(mol)
img=Draw.MolsToGridImage(mols,molsPerRow=4,subImgSize=(200,200),legends=['' for x in mols])
img

from rdkit import Chem
from rdkit.Chem import  Draw
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
opts =  DrawingOptions()
opts.includeAtomNumbers=True
m = Chem.MolFromSmiles('OC1C2C1CC2')
opts.includeAtomNumbers=True
opts.bondLineWidth=2.8
Draw.MolToImage(m,options=opts)

from rdkit import Chem


for atom in mol.GetAtoms():
    atom.SetProp('atomLabel',str(atom.GetIdx()))
mol


m = Chem.MolFromSmiles('c1ccccc1OC')
patt = Chem.MolFromSmarts('OC')
flag =m.HasSubstructMatch(patt)
if flag:
    print("molecu m contains group -OCH3")
else:
    print("molecu m don't contain group -OCH3")
for atom in m.GetAtoms():
    atom.SetProp('atomLabel',str(atom.GetIdx()))
m
    

m = Chem.MolFromSmiles('c1ccccc1OC')
patt = Chem.MolFromSmarts('OC')
flag =m.HasSubstructMatch(patt)
if flag:
    atomids = m.GetSubstructMatch(patt)
    print("matched atom id:",atomids)
else:
    print("molecu m don't contain group -OCH3")

m = Chem.MolFromSmiles('c1ccc(OC)cc1OC')
patt = Chem.MolFromSmarts('OC')
flag =m.HasSubstructMatch(patt)
if flag:
    atomids = m.GetSubstructMatches(patt)
    print("matched atom id:",atomids)
else:
    print("molecu m don't contain group -OCH3")


m = Chem.MolFromSmiles('CC[C@H](F)Cl')
print(m.HasSubstructMatch(Chem.MolFromSmiles('C[C@H](F)Cl')))
print(m.HasSubstructMatch(Chem.MolFromSmiles('C[C@@H](F)Cl')))
print(m.HasSubstructMatch(Chem.MolFromSmiles('CC(F)Cl')))
m = Chem.MolFromSmiles('CC[C@H](F)Cl')
a=m.HasSubstructMatch(Chem.MolFromSmiles('C[C@H](F)Cl'),useChirality=True)
b=m.HasSubstructMatch(Chem.MolFromSmiles('C[C@@H](F)Cl'),useChirality=True)
c=m.HasSubstructMatch(Chem.MolFromSmiles('CC(F)Cl'),useChirality=True)
print(a)
print(b)
print(c)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles('COc1c(Br)cccc1OC')
patt = Chem.MolFromSmarts('OC')
repsmis= ['F','Cl','Br','O']
mols=[]
mols.append(m)
for r in repsmis:
    rep = Chem.MolFromSmarts(r)
    res = AllChem.ReplaceSubstructs(m,patt,rep)
    mols.extend(res)
# 为了标准化smiles,可以将得到的分子mol-》smiles->mol,然后对其可视化。
smis = [ Chem.MolToSmiles(mol)      for mol in mols]
mols = [Chem.MolFromSmiles(smi)  for smi in smis]
Draw.MolsToGridImage(mols,molsPerRow=3,subImgSize=(200,200),legends=['' for x in mols])

m1 = Chem.MolFromSmiles('BrCCc1cncnc1C(=O)O')
core = Chem.MolFromSmiles('c1cncnc1')
tmp = Chem.ReplaceSidechains(m1,core)
m1
tmp

m1 = Chem.MolFromSmiles('BrCCc1cncnc1C(=O)O')
core = Chem.MolFromSmiles('c1cncnc1')
tmp = Chem.ReplaceCore(m1,core)
tmp

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles('COc1c(Br)cccc1OC')
core = Chem.MolFromSmiles('c1c(Br)cccc1')
core_v = Chem.ReplaceSidechains(m,core)
core_v


from rdkit.Chem import rdFMCS
mol1 = Chem.MolFromSmiles("O=C(NCc1cc(OC)c(O)cc1)CCCC/C=C/C(C)C")
mol2 = Chem.MolFromSmiles("CC(C)CCCCCC(=O)NCC1=CC(=C(C=C1)O)OC")
mol3 = Chem.MolFromSmiles("c1(C=O)cc(OC)c(O)cc1")
mols = [mol1,mol2,mol3]
res=rdFMCS.FindMCS(mols)
res
res.numAtoms

res.numBonds

res.smartsString
aa= Chem.MolFromSmarts(res.smartsString)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
m1 = Chem.MolFromSmiles('C=CC(=O)N1CCC(CC1)C2CCNC3=C(C(=NN23)C4=CC=C(C=C4)OC5=CC=CC=C5)C(=O)N')
m2 = Chem.MolFromSmiles('CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4')
m3= Chem.MolFromSmiles('CNC1(CCCCC1=O)C1=CC=CC=C1Cl')
core_m1 = MurckoScaffold.GetScaffoldForMol(m1)
core_m2 = MurckoScaffold.GetScaffoldForMol(m2)
core_m3 = MurckoScaffold.GetScaffoldForMol(m3)
core_mols=[core_m1,core_m2,core_m3]
Draw.MolsToGridImage(core_mols,molsPerRow=3,subImgSize=(300,300),legends=['' for x in core_mols])

from rdkit.Chem import AllChem

def getrxns(rxn,productsmi):
    productmol = Chem.MolFromSmiles(productsmi)
    reactions = rxn.RunReactants([productmol])
    rxns = []
    for reaction in reactions:
        smis=[]
        for compound in reaction:
            smi = Chem.MolToSmiles(compound)
            smis.append(smi)

        rxnstr='.'.join(smis)+'>>'+productsmi
        newr=canon_reaction(rxnstr)
        rxns.append(newr)
    return rxns


tem='([Cl;H0;D1;+0:1]-[c;H0;D3;+0:2](:[c:3]):[n;H0;D2;+0:4]:[c:5])>>(C-[n;H0;D3;+0:4](:[c:5]):[c;H0;D3;+0:2](=O):[c:3]).(Cl-P(-Cl)(=O)-[Cl;H0;D1;+0:1])'
rxn = AllChem.ReactionFromSmarts(tem)
productsmi='CC(C)(Nc1nc(Cl)c(-c2ccc(F)cc2)c(-c2ccncc2)n1)c1ccccc1'
reactions =getrxns(rxn,products_smi[0])
for reaction in reactions:
    img=ReactionStringToImage(reaction)
    display(img)

from rdkit.Chem import AllChem
tem='([Cl;H0;D1;+0:1]-[c;H0;D3;+0:2](:[c:3]):[n;H0;D2;+0:4]:[c:5])>>(C-[n;H0;D3;+0:4](:[c:5]):[c;H0;D3;+0:2](=O):[c:3]).(Cl-P(-Cl)(=O)-[Cl;H0;D1;+0:1])'
rxn = AllChem.ReactionFromSmarts(tem)
def getrxns_reactants(rxn,productsmi):
    productmol = Chem.MolFromSmiles(productsmi)
    reactions = rxn.RunReactants([productmol])
    rxns = []
    for reaction in reactions:
        smis=[]
        for compound in reaction:
#             display(compound)
            smi = Chem.MolToSmiles(compound)
            smis.append(smi)

        newr='.'.join(smis)
        rxns.append(newr)
    return rxns

prosmi="COC(=O)c1cccc(-c2nc(Cl)cc3c(OC)nc(C(C)C)n23)c1"
# prosmi='CC(C)(Nc1nc(Cl)c(-c2ccc(F)cc2)c(-c2ccncc2)n1)c1ccccc1'
rs=getrxns_reactants(rxn,prosmi)


smi=rs[0]
m = Chem.MolFromSmiles(smi,sanitize=False)
if m is None:
    print('invalid SMILES')
else:
    try:
        Chem.SanitizeMol(m)
        print("smiles is ok")
    except:
        print('invalid chemistry')
        
        


from rdkit.Chem import Draw
rxn = AllChem.ReactionFromSmarts('[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])O.[N-:13]=[N+:14]=[N-:15]>C(Cl)Cl.C(=O)(C(=O)Cl)Cl>[cH:5]1[cH:6][c:7]2[cH:8][n:9][cH:10][cH:11][c:12]2[c:3]([cH:4]1)[C:2](=[O:1])[N:13]=[N+:14]=[N-:15]',useSmiles=True)
d2d = Draw.MolDraw2DCairo(800,300)
d2d.DrawReaction(rxn)
png = d2d.GetDrawingText()
#open('./reaction1.o.png','wb+').write(png)  
d2d = Draw.MolDraw2DCairo(800,300)
d2d.DrawReaction(rxn,highlightByReactant=True)


from rdkit import DataStructs
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
smis=[
    'CC(=O)CC(C1=CC=C(C=C1)[N+]([O-])=O)C1=C(O)C2=CC=CC=C2OC1=O',
'CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2',
'CCC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2'
]
mols =[]
for smi in smis:
    m = Chem.MolFromSmiles(smi)
    mols.append(m)

fps = [Chem.RDKFingerprint(x) for x in mols]
sm01=DataStructs.FingerprintSimilarity(fps[0],fps[1])

sm02=DataStructs.FingerprintSimilarity(fps[0],fps[2])

sm12=DataStructs.FingerprintSimilarity(fps[1],fps[2])
print("similarity between mol 1 and mol2: %.2f"%sm01)
print("similarity between mol 1 and mol3: %.2f"%sm02)
print("similarity between mol 2 and mol3: %.2f"%sm12)


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
# Pictet-Spengler rxn
rxn = AllChem.ReactionFromSmarts('[cH1:1]1:[c:2](-[CH2:7]-[CH2:8]-[NH2:9]):[c:3]:[c:4]:[c:5]:[c:6]:1.[#6:11]-[CH1;R0:10]=[OD1]>>[c:1]12:[c:2](-[CH2:7]-[CH2:8]-[NH1:9]-[C:10]-2(-[#6:11])):[c:3]:[c:4]:[c:5]:[c:6]:1')
Draw.ReactionToImage(rxn)
rxnn=AllChem.ReactionFromSmarts('([#8&v2:1]=[#6&v4:2](-[#6&v4:3](-[#6&v4:4])(-[#1&v1:5])-[#1&v1:6])-[#6&v4:7](=[#8&v2:8])-[#8&v2:9])>>([#7&v3](-[#6&v4:2](-[#6&v4:3](-[#6&v4:4])(-[#1&v1:5])-[#1&v1:6])(-[#6&v4:7](=[#8&v2:8])-[#8&v2:9])-[#1&v1])(-[#1&v1])-[#1&v1].[#8&v2](-[#6&v4](=[#8&v2])-[#6&v4](=[#8&v2:1])-[#6&v4](-[#6&v4](-[#6&v4](=[#8&v2])-[#8&v2]-[#1&v1])(-[#1&v1])-[#1&v1])(-[#1&v1])-[#1&v1])-[#1&v1])')
Draw.ReactionToImage(rxnn)
rxn16=AllChem.ReactionFromSmarts('([#8&v2:1]=[#6&v4:2](-[#6&v4:3](-[#6&v4:4](-[#8&v2:5]-[#1&v1:6])=[#8&v2:7])(-[#1&v1:8])-[#1&v1:9])-[#6&v4:10](=[#8&v2:11])-[#8&v2:12]-[#1&v1:13])>>([#7&v3](-[#6&v4:2](-[#6&v4:3](-[#6&v4:4](-[#8&v2:5]-[#1&v1:6])=[#8&v2:7])(-[#1&v1:8])-[#1&v1:9])(-[#6&v4:10](=[#8&v2:11])-[#8&v2:12]-[#1&v1:13])-[#1&v1])(-[#1&v1])-[#1&v1].[#8&v2](-[#6&v4](=[#8&v2])-[#6&v4](=[#8&v2:1])-[#6&v4](-[#6&v4](-[#6&v4](=[#8&v2])-[#8&v2]-[#1&v1])(-[#1&v1])-[#1&v1])(-[#1&v1])-[#1&v1])-[#1&v1])')
Draw.ReactionToImage(rxn16)
rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])-[OD1].[N!H0:3]>>[C:1](=[O:2])[N:3]')
Draw.ReactionToImage(rxn)
s1='[C:1](=[O:2])-[OD1].[N!H0:3]'
s2='[C:1](=[O:2])[N:3]'
Draw.MolToImage(Chem.MolFromSmarts(s1))
Draw.MolToImage(Chem.MolFromSmarts(s2))
s1=Chem.MolFromSmiles('CC(=O)CC(C1=CC=C(C=C1)[N+]([O-])=O)C1=C(O)C2=CC=CC=C2OC1=O')
s2=Chem.MolFromSmiles('CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2')
Draw.MolToImage(s1)
Draw.MolToImage(s2)
rxnn=AllChem.ReactionFromSmarts('CC(=O)CC(C1=CC=C(C=C1)[N+]([O-])=O)C1=C(O)C2=CC=CC=C2OC1=O>>CC(=O)CC(C1=CC=CC=C1)C1=C(O)C2=C(OC1=O)C=CC=C2')
Draw.ReactionToImage(rxnn)




from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem import DataStructs
# construct the chemical reactions
rxn1 = rdChemReactions.ReactionFromSmarts('CCCO>>CCC=O')
rxn2 = rdChemReactions.ReactionFromSmarts('CC(O)C>>CC(=O)C')
rxn3 = rdChemReactions.ReactionFromSmarts('NCCO>>NCC=O')

# construct difference fingerprint (subtracts reactant fingerprint from product)
fp1 = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn1)
fp2 = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn2)
fp3 = rdChemReactions.CreateDifferenceFingerprintForReaction(rxn3)

print(DataStructs.TanimotoSimilarity(fp1,fp2))

# The similarity between fp1 and fp2 is zero because as far as the reaction
# fingerprint is concerned, the parts which change within the reactions have
# nothing in common with each other.
# In contrast, fp1 and fp3 have some common parts
print(DataStructs.TanimotoSimilarity(fp1,fp3))

fpn=rdChemReactions.CreateDifferenceFingerprintForReaction(rxnn)
fpn16=rdChemReactions.CreateDifferenceFingerprintForReaction(rxn16)
frxn=rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
print(DataStructs.TanimotoSimilarity(fpn,fpn16))
print(DataStructs.TanimotoSimilarity(fpn,frxn))
print(DataStructs.TanimotoSimilarity(frxn,fpn16))


