from __future__ import annotations

import anndata as ad
import scanpy as sc


def preprocess_single_cell(
    adata: ad.AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    max_pct_mt: float = 20.0,
    target_sum: float = 1e4,
    n_top_genes: int = 3000,
    hvg_flavor: str = "seurat",
    inplace: bool = False,
    mt_gene_prefix: str = "MT-",
) -> ad.AnnData:
    """Run a standard Scanpy preprocessing workflow for scRNA-seq data.

    Steps:
    1. Basic QC metrics (`total_counts`, `n_genes_by_counts`, mitochondrial ratio)
    2. Cell and gene filtering
    3. Library-size normalization + log1p
    4. Highly variable gene (HVG) selection and subsetting

    Parameters
    ----------
    adata
        Input AnnData object with raw counts in `adata.X`.
    min_genes
        Minimum detected genes per cell.
    min_cells
        Minimum cells expressing a gene.
    max_pct_mt
        Maximum mitochondrial percentage allowed per cell.
    target_sum
        Target total counts for normalization (`sc.pp.normalize_total`).
    n_top_genes
        Number of HVGs to keep.
    hvg_flavor
        Method used by `sc.pp.highly_variable_genes`.
    inplace
        If False, preprocesses a copy and returns it.
    mt_gene_prefix
        Prefix used to identify mitochondrial genes in `adata.var_names`.

    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    if adata is None:
        raise ValueError("`adata` cannot be None.")

    obj = adata if inplace else adata.copy()

    # 1) QC metrics
    obj.var["mt"] = obj.var_names.str.upper().str.startswith(mt_gene_prefix.upper())
    sc.pp.calculate_qc_metrics(obj, qc_vars=["mt"], inplace=True)

    # 2) Cell/gene filtering
    sc.pp.filter_cells(obj, min_genes=min_genes)
    obj = obj[obj.obs["pct_counts_mt"] < max_pct_mt].copy()
    sc.pp.filter_genes(obj, min_cells=min_cells)

    # 3) Normalize and log-transform
    sc.pp.normalize_total(obj, target_sum=target_sum)
    sc.pp.log1p(obj)

    # 4) HVG selection and subsetting
    sc.pp.highly_variable_genes(obj, n_top_genes=n_top_genes, flavor=hvg_flavor)
    if "highly_variable" not in obj.var.columns:
        raise RuntimeError("HVG selection failed: `highly_variable` flag not found in `obj.var`.")
    obj = obj[:, obj.var["highly_variable"]].copy()

    return obj


__all__ = ["preprocess_single_cell"]
