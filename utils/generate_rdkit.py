import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import rdDetermineBonds


def add_bonds_custom(mol, coords, delta=0.25):
    ptable = Chem.GetPeriodicTable()
    n = mol.GetNumAtoms()
    for i in range(n):
        zi = mol.GetAtomWithIdx(i).GetAtomicNum()
        for j in range(i + 1, n):
            zj = mol.GetAtomWithIdx(j).GetAtomicNum()
            dij = np.linalg.norm(coords[i] - coords[j])
            cutoff = ptable.GetRcovalent(zi) + ptable.GetRcovalent(zj) + delta
            if dij < cutoff:
                mol.AddBond(i, j, rdchem.BondType.SINGLE)

from rdkit.Geometry import Point3D
def get_smiles(atomic_numbers=None, coords=None):
    rw = Chem.RWMol()
    if atomic_numbers is None:
        atomic_numbers = []
    if coords is None:
        coords = np.empty((0, 3), dtype=float)

    for z in atomic_numbers:
        rw.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(len(atomic_numbers))  # ⬅︎ 新增
    for i, (x, y, z) in enumerate(coords):  # ⬅︎ 新增
        conf.SetAtomPosition(i, Point3D(float(x),  # ⬅︎ 新增
                                        float(y),
                                        float(z)))
    rw.AddConformer(conf, assignId=True)

    try:
        mol_tmp = rw.GetMol()
        rdDetermineBonds.DetermineBonds(mol_tmp, charge=-1)
        Chem.SanitizeMol(mol_tmp)
        mol = mol_tmp
    except ValueError:
        # 将已存在的键逐个删除后，再自定义加键，避免调用 RemoveAllBonds()
        bonds = list(rw.GetBonds())
        for bond in reversed(bonds):
            rw.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        add_bonds_custom(rw, coords)
        mol = rw.GetMol()

    # Chem.AssignAtomChiralTagsFromStructure(mol)
    # Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    print("生成的 SMILES:", smiles)
    return smiles, mol

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_pos(smile, mol_ref):
    # ----- 1. 输入 SMILES -------
    smiles = smile          # 示例：乙醇；改成你的 SMILES

    # ----- 2. 构建拓扑并加氢 ----
    # mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # print("原子数(含显式氢):", mol.GetNumAtoms()) # 会打印 9
    # flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    # Chem.SanitizeMol(mol, sanitizeOps=flags)
    #
    # # 直接嵌入，不再调用 AddHs()
    # params = AllChem.ETKDGv3()
    # params.randomSeed = 42
    # cid = AllChem.EmbedMolecule(mol, params)
    # AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol, explicitOnly=True, addCoords=True)

        # ----- 3. 生成 3D 构象 -------
        params = AllChem.ETKDGv3()  # ETKDG v3 推荐参数
        params.randomSeed = 42  # 固定随机种子，结果可复现
        cid = AllChem.EmbedMolecule(mol, params)  # 若返回 -1 表示失败
        if mol.GetNumConformers() == 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            # params.maxIters = 100
            cid = AllChem.EmbedMolecule(mol, params)
            if cid == -1:  # 仍失败 → 尝试随机坐标
                params.useRandomCoords = True
                cid = AllChem.EmbedMolecule(mol, params)
            if cid == -1:
                print('fail')
                return None, None, None

        else:
            cid = 0
            # （可选）UFF 最小化 200 次迭代
        AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=200)

        # mol_reordered = match(mol, mol_ref)
        aligned_prb, heavy_rms, mapping = heavy_rmsd(mol_ref, mol)

        mol_reordered = aligned_prb

        # ----- 4. 提取原子序数和坐标 ----
        conf = mol_reordered.GetConformer(cid)
        N = mol_reordered.GetNumAtoms()

        atomic_numbers = [atom.GetAtomicNum() for atom in mol_reordered.GetAtoms()]
        positions = np.array([[conf.GetAtomPosition(i).x,
                               conf.GetAtomPosition(i).y,
                               conf.GetAtomPosition(i).z] for i in range(N)],
                             dtype=float)

        print("原子序号:", atomic_numbers)
        print("坐标矩阵 shape:", positions.shape)
        print(positions)

        return positions, atomic_numbers, mapping
    except Exception as exc:  # 捕获所有异常
        print(f"[safe_get_smiles] 生成失败: {exc}")
        return None, None, None

# 若需要保存 numpy 文件：
# np.save("positions.npy", positions)
# np.save("atomic_numbers.npy", np.array(atomic_numbers, dtype=int))

from rdkit.Chem import rdMolAlign
def match(mol_rdkit, mol_ref):


    # mapping 将被函数填充；元素是 (ref_idx, probe_idx) 对
    mapping = []
    rms = rdMolAlign.GetBestRMS(mol_rdkit,      # probe (要重排的那个)
        mol_ref,        # reference (精确构象)
        prbId=0,        # conformer id
        refId=0,
        map=mapping)

    print("最小 RMSD =", rms, "Å")
    print("原子对应列表:", mapping[:10], " ...")  # [(0,2), (1,1), ...]

    # 建立长度 N 的列表，位置 = ref_idx，值 = 对应的 probe_idx
    N = mol_ref.GetNumAtoms()
    new_order = [None] * N
    for ref_idx, probe_idx in mapping:
        new_order[ref_idx] = probe_idx

    assert None not in new_order, "映射不完整，可能含对称体或缺H"
    print("新顺序（probe 原子索引）:", new_order)
    from rdkit import Chem

    # RenumberAtoms 接受 probe→new_index 的列表
    mol_reordered = Chem.RenumberAtoms(mol_rdkit, newOrder=new_order)

    # 再次对齐验证
    rms2 = rdMolAlign.AlignMol(mol_reordered, mol_ref, prbCid=0, refCid=0)
    print("重新排序后 RMSD =", rms2, "Å")  # 应与第一步同值

    return mol_reordered

    # conf = mol_reordered.GetConformer(0)
    # coords = np.array([[conf.GetAtomPosition(i).x,
    #                     conf.GetAtomPosition(i).y,
    #                     conf.GetAtomPosition(i).z]
    #                    for i in range(mol_reordered.GetNumAtoms())])
    #
    # atomic_numbers = [a.GetAtomicNum() for a in mol_reordered.GetAtoms()]

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

def heavy_rmsd(mol_ref_full, mol_prb_full, confId_ref=0, confId_prb=0):
    # 1) 生成仅重原子的副本
    # heavy_ref = Chem.RemoveHs(mol_ref_full, implicitOnly=False)
    # heavy_prb = Chem.RemoveHs(mol_prb_full, implicitOnly=False)

    # heavy_ref = mol_ref_full
    # heavy_prb = mol_prb_full
    #
    # # 2) 找最优重原子映射 + RMSD   (注意这里传入 heavy 副本)
    # heavy_map = []                                 # 写入 (refHeavyIdx , prbHeavyIdx)
    # rmsd = rdMolAlign.GetBestRMS(
    #     heavy_prb, heavy_ref,
    #     prbId=confId_prb, refId=confId_ref,
    #     map=heavy_map)
    #
    # # 3)  把 heavy 索引转换成“完整 mol”里的索引 ----------------
    # heavy_idx_ref = [i for i,a in enumerate(mol_ref_full.GetAtoms()) if a.GetAtomicNum()>1]
    # heavy_idx_prb = [i for i,a in enumerate(mol_prb_full.GetAtoms()) if a.GetAtomicNum()>1]
    #
    # atom_map_full = [(heavy_idx_ref[r], heavy_idx_prb[p])       # refFullIdx , prbFullIdx
    #                  for (r, p) in heavy_map]
    # # ---------------------------------------------------------
    #
    # # 4) 用完整 mol + full 映射做刚体对齐（氢跟着动）
    # rdMolAlign.AlignMol(
    #     prbMol=mol_prb_full,
    #     refMol=mol_ref_full,
    #     prbCid=confId_prb,
    #     refCid=confId_ref,
    #     atomMap=heavy_map)
    #
    # return mol_prb_full, rmsd, heavy_map

    mol_prb = mol_prb_full
    mol_ref = mol_ref_full

    # atom_map = []  # ← 关键：用 list 接收映射
    # rmsd = rdMolAlign.GetBestRMS(
    #     prbMol=mol_prb,
    #     refMol=mol_ref,
    #     prbId=confId_prb,
    #     refId=confId_ref,
    #     map=atom_map)

    # 2) 在同一映射下做刚体对齐（氢自动随重原子一起动）
    atom_map = []
    min_rmsd = 0
    rdMolAlign.AlignMol(
        prbMol=mol_prb,
        refMol=mol_ref,
        prbCid=confId_prb,
        refCid=confId_ref)
    print("Atoms:", mol_ref.GetNumAtoms(), mol_prb.GetNumAtoms())
    print("Formal charge:", Chem.GetFormalCharge(mol_ref),
          Chem.GetFormalCharge(mol_prb))
    print("SMILES:", Chem.MolToSmiles(mol_ref),
          Chem.MolToSmiles(mol_prb))
    print("Conformers:", mol_ref.GetNumConformers(),
          mol_prb.GetNumConformers())

    return mol_prb, min_rmsd, atom_map

# # -------- 示例 --------
# sm = "CC(O)C"          # 任意示例 SMILES
# # 生成参考、probe 两个不同构象
# mol_ref  = Chem.AddHs(Chem.MolFromSmiles(sm))
# mol_prb  = Chem.AddHs(Chem.MolFromSmiles(sm))
# AllChem.EmbedMolecule(mol_ref, AllChem.ETKDG())
# AllChem.EmbedMolecule(mol_prb, AllChem.ETKDG())
# AllChem.UFFOptimizeMolecule(mol_prb)          # 故意让它与 ref 不同
#
# aligned_prb, heavy_rms, mapping = heavy_rmsd(mol_ref, mol_prb)
# print("最小重原子 RMSD =", heavy_rms, "Å")
# print("原子对应关系 (前几个):", mapping[:5])


if __name__ == "__main__":
    atomic_numbers = [6, 6, 8, 1, 1, 1, 1]
    coords = np.array([
        [0.000, 0.000, 0.000],
        [1.522, 0.000, 0.000],
        [2.081, 1.207, 0.000],
        [-0.543, 0.934, 0.000],
        [-0.543, -0.467, 0.890],
        [-0.543, -0.467, -0.890],
        [2.640, 0.000, 0.000],
    ], dtype=float)
    smiles, mol = get_smiles(atomic_numbers=atomic_numbers, coords=coords)
    generate_pos(smiles, mol)
    # generate_pos("[H]C(NC([H])([H])C(O)O)N([H])[H]")