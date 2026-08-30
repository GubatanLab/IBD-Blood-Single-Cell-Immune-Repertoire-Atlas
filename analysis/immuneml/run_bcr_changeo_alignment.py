#!/usr/bin/env python3
"""Align the exported BCR AIRR dataset with IgBLAST/Change-O.

This script uses OGRDB human IG germline tables that were downloaded into
``chain_bcr_immuneml/changeo_alignment/references`` and the local NCBI IgBLAST
Windows bundle under ``chain_bcr_immuneml/ncbi-igblast-1.22.0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "chain_bcr_immuneml"
DEFAULT_UNALIGNED = DEFAULT_BASE / "bcr_shm_immcantation_unaligned_airr.tsv"
DEFAULT_OUT = DEFAULT_BASE / "changeo_alignment"
DEFAULT_IGBLAST = DEFAULT_BASE / "ncbi-igblast-1.22.0"
DEFAULT_PYTHON = Path(r"C:/path/to/private-python-environment\python.exe")
DEFAULT_SCRIPTS = Path(r"C:/path/to/private-python-environment\Scripts")


def run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> None:
    if quiet:
        label = Path(cmd[0]).name
        query_label = Path(cmd[cmd.index("-query") + 1]).name if "-query" in cmd else ""
        print(f"\n[run quiet] {label} {query_label}", flush=True)
        subprocess.run(
            cmd,
            cwd=str(cwd or ROOT),
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print("\n[run]", " ".join(str(x) for x in cmd), flush=True)
        subprocess.run(cmd, cwd=str(cwd or ROOT), check=True, env=env)


def clean_nt(seq: str, keep_imgt_gaps: bool = False) -> str:
    seq = (seq or "").upper().replace(" ", "").replace("\r", "").replace("\n", "")
    allowed = "ACGTN." if keep_imgt_gaps else "ACGTN"
    return "".join(ch for ch in seq if ch in allowed)


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")


def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    seq_parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        records.append((name, "".join(seq_parts)))
    return records


def split_fasta_by_locus(
    path: Path,
    chunk_dir: Path,
    chunk_size: int,
    metadata: dict[str, dict[str, str]],
) -> list[tuple[str, Path]]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    records = read_fasta_records(path)
    by_locus: dict[str, list[tuple[str, str]]] = {"IGH": [], "IGK": [], "IGL": []}
    for name, seq in records:
        locus = metadata.get(name, {}).get("locus", "")
        if locus in by_locus:
            by_locus[locus].append((name, seq))

    chunks: list[tuple[str, Path]] = []
    for locus, locus_records in by_locus.items():
        for start in range(0, len(locus_records), chunk_size):
            chunk_index = (start // chunk_size) + 1
            chunk_path = chunk_dir / f"bcr_airr_{locus}_chunk_{chunk_index:04d}.fasta"
            write_fasta(locus_records[start : start + chunk_size], chunk_path)
            chunks.append((locus, chunk_path))
    return chunks


def build_references(ref_dir: Path, out_dir: Path, include_inferred: bool) -> dict[str, Path]:
    groups: dict[str, list[tuple[str, str]]] = {"v": [], "d": [], "j": [], "c": []}
    locus_groups: dict[str, list[tuple[str, str]]] = {
        "igh_v": [],
        "igh_d": [],
        "igh_j": [],
        "igk_v": [],
        "igk_j": [],
        "igl_v": [],
        "igl_j": [],
    }
    germline_records: list[tuple[str, str]] = []
    summary: dict[str, dict[str, int]] = {}

    for csv_path in sorted(ref_dir.glob("ogrdb_human_*_all.csv")):
        locus_summary = {"input_rows": 0, "kept_rows": 0, "skipped_rows": 0}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                locus_summary["input_rows"] += 1
                gene = row.get("Gene", "")
                allele = row.get("Allele", "")
                if not gene or not allele:
                    locus_summary["skipped_rows"] += 1
                    continue
                if row.get("In_Published_Set") != "Yes":
                    locus_summary["skipped_rows"] += 1
                    continue
                if not include_inferred and "*i" in allele:
                    locus_summary["skipped_rows"] += 1
                    continue

                if re.match(r"^IG[HKL]V", gene):
                    group = "v"
                elif re.match(r"^IGHD", gene):
                    group = "d"
                elif re.match(r"^IG[HKL]J", gene):
                    group = "j"
                elif re.match(r"^IGH[AEGM]", gene):
                    group = "c"
                else:
                    locus_summary["skipped_rows"] += 1
                    continue

                blast_seq = clean_nt(row.get("Sequence", ""), keep_imgt_gaps=False)
                if len(blast_seq) < 10:
                    locus_summary["skipped_rows"] += 1
                    continue
                groups[group].append((allele, blast_seq))
                locus = gene[:3].lower()
                locus_key = f"{locus}_{group}"
                if locus_key in locus_groups:
                    locus_groups[locus_key].append((allele, blast_seq))

                if group == "v":
                    germ_seq = clean_nt(row.get("Gapped_Sequence", ""), keep_imgt_gaps=True)
                    if not germ_seq:
                        germ_seq = blast_seq
                else:
                    germ_seq = blast_seq
                germline_records.append((allele, germ_seq))
                locus_summary["kept_rows"] += 1
        summary[csv_path.name] = locus_summary

    ref_out = out_dir / "references"
    paths = {
        "v": ref_out / "imgt_human_ig_v.fasta",
        "d": ref_out / "imgt_human_ig_d.fasta",
        "j": ref_out / "imgt_human_ig_j.fasta",
        "c": ref_out / "imgt_human_ig_c.fasta",
        "dummy_d": ref_out / "imgt_human_dummy_d.fasta",
        "germline": ref_out / "ogrdb_human_ig_changeo_germline.fasta",
    }
    for key in locus_groups:
        paths[key] = ref_out / f"imgt_human_{key}.fasta"
    for group in ("v", "d", "j", "c"):
        write_fasta(groups[group], paths[group])
    for key, records in locus_groups.items():
        write_fasta(records, paths[key])
    write_fasta([("dummyD*01", "NNNNNNNNNNNNNNN")], paths["dummy_d"])
    write_fasta(germline_records, paths["germline"])

    summary["fasta_counts"] = {group: len(records) for group, records in groups.items()}
    summary["locus_fasta_counts"] = {group: len(records) for group, records in locus_groups.items()}
    summary["fasta_counts"]["germline"] = len(germline_records)
    (out_dir / "reference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return paths


def export_query_fasta(airr_path: Path, fasta_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    records: list[tuple[str, str]] = []
    with airr_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq_id = row["sequence_id"]
            seq = clean_nt(row.get("sequence", ""), keep_imgt_gaps=False)
            if len(seq) < 50:
                continue
            records.append((seq_id, seq))
            metadata[seq_id] = row
    write_fasta(records, fasta_path)
    return metadata


def prepare_igdata(igblast_dir: Path, out_dir: Path) -> Path:
    igdata = out_dir / "igblast_data"
    db_dir = igdata / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(igblast_dir / "internal_data", igdata / "internal_data", dirs_exist_ok=True)
    shutil.copytree(igblast_dir / "optional_file", igdata / "optional_file", dirs_exist_ok=True)
    return igdata


def make_blast_db(makeblastdb: Path, fasta: Path, db_prefix: Path) -> None:
    run([
        str(makeblastdb),
        "-dbtype",
        "nucl",
        "-parse_seqids",
        "-in",
        str(fasta),
        "-out",
        str(db_prefix),
    ])


def concatenate_text(files: list[Path], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for file in files:
            with file.open(encoding="utf-8", errors="replace") as inp:
                shutil.copyfileobj(inp, out)


def fmt7_query_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("# Query:"):
                count += 1
    return count


def build_reference_index(paths: dict[str, Path]) -> dict[str, dict[str, set[str]]]:
    index: dict[str, dict[str, set[str]]] = {}
    for key in ("igh_v", "igh_d", "igh_j", "igk_v", "igk_j", "igl_v", "igl_j"):
        ids = {name for name, _ in read_fasta_records(paths[key])}
        by_gene: dict[str, set[str]] = {}
        for seq_id in ids:
            gene = seq_id.split("*", 1)[0]
            by_gene.setdefault(gene, set()).add(seq_id)
        index[key] = {"ids": ids, "by_gene": by_gene}
    return index


def normalize_call(value: str) -> list[str]:
    calls: list[str] = []
    for part in re.split(r"[,;|]", value or ""):
        part = part.strip()
        if not part or part.upper() == "NA":
            continue
        calls.append(part)
    return calls


def write_seqid_list_for_chunk(
    chunk: Path,
    locus: str,
    segment: str,
    metadata: dict[str, dict[str, str]],
    ref_index: dict[str, dict[str, set[str]]],
) -> Path | None:
    field = {"v": "v_call", "d": "d_call", "j": "j_call"}[segment]
    ref_key = f"{locus.lower()}_{segment}"
    if ref_key not in ref_index:
        return None
    available = ref_index[ref_key]["ids"]
    by_gene = ref_index[ref_key]["by_gene"]
    selected: set[str] = set()
    for seq_id, _ in read_fasta_records(chunk):
        for call in normalize_call(metadata.get(seq_id, {}).get(field, "")):
            if call in available:
                selected.add(call)
            else:
                selected.update(by_gene.get(call.split("*", 1)[0], set()))
    if not selected:
        return None
    list_path = chunk.with_suffix(f".{segment}_seqids.txt")
    list_path.write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    return list_path


def run_igblast_chunk(
    igblastn: Path,
    locus: str,
    chunk: Path,
    out_file: Path,
    out_dir: Path,
    igblast_threads: int,
    metadata: dict[str, dict[str, str]],
    ref_index: dict[str, dict[str, set[str]]],
    resume_existing: bool,
) -> Path:
    expected_queries = len(read_fasta_records(chunk))
    if resume_existing and fmt7_query_count(out_file) == expected_queries:
        print(f"\n[skip] {out_file.name} already has {expected_queries} queries", flush=True)
        return out_file

    env = dict(os.environ)
    env["IGDATA"] = str(out_dir / "igblast_data")
    locus_lower = locus.lower()
    d_db = "imgt_human_igh_d" if locus == "IGH" else "imgt_human_dummy_d"
    cmd = [
        str(igblastn),
        "-query",
        str(chunk),
        "-germline_db_V",
        str(out_dir / "igblast_data" / "database" / f"imgt_human_{locus_lower}_v"),
        "-germline_db_D",
        str(out_dir / "igblast_data" / "database" / d_db),
        "-germline_db_J",
        str(out_dir / "igblast_data" / "database" / f"imgt_human_{locus_lower}_j"),
        "-organism",
        "human",
        "-ig_seqtype",
        "Ig",
        "-auxiliary_data",
        str(out_dir / "igblast_data" / "optional_file" / "human_gl.aux"),
        "-domain_system",
        "imgt",
        "-outfmt",
        "7 std qseq sseq btop",
        "-strand",
        "plus",
        "-num_alignments_V",
        "1",
        "-num_alignments_D",
        "1",
        "-num_alignments_J",
        "1",
        "-num_threads",
        str(igblast_threads),
        "-out",
        str(out_file),
    ]
    v_list = write_seqid_list_for_chunk(chunk, locus, "v", metadata, ref_index)
    d_list = write_seqid_list_for_chunk(chunk, locus, "d", metadata, ref_index) if locus == "IGH" else None
    j_list = write_seqid_list_for_chunk(chunk, locus, "j", metadata, ref_index)
    if v_list is not None:
        cmd.extend(["-germline_db_V_seqidlist", str(v_list)])
    if d_list is not None:
        cmd.extend(["-germline_db_D_seqidlist", str(d_list)])
    if j_list is not None:
        cmd.extend(["-germline_db_J_seqidlist", str(j_list)])
    run(cmd, env=env, quiet=True)
    return out_file


def find_one(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {path}")
    return matches[0]


def merge_metadata(germline_airr: Path, metadata: dict[str, dict[str, str]], out_path: Path) -> int:
    with germline_airr.open(newline="", encoding="utf-8") as inp:
        reader = csv.DictReader(inp, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        additions = ["Sample", "SampleID", "PatientID", "Diagnosis1", "duplicate_count", "input_locus"]
        for field in additions:
            if field not in fieldnames:
                fieldnames.append(field)
        rows = []
        for row in reader:
            meta = metadata.get(row.get("sequence_id", ""), {})
            row["Sample"] = meta.get("Sample", row.get("Sample", ""))
            row["SampleID"] = meta.get("SampleID", row.get("SampleID", ""))
            row["PatientID"] = meta.get("PatientID", row.get("PatientID", ""))
            row["Diagnosis1"] = meta.get("Diagnosis1", row.get("Diagnosis1", ""))
            row["duplicate_count"] = meta.get("duplicate_count", row.get("duplicate_count", "1"))
            row["input_locus"] = meta.get("locus", row.get("input_locus", ""))
            if "locus" not in row or not row.get("locus"):
                row["locus"] = meta.get("locus", row.get("locus", ""))
                if "locus" not in fieldnames:
                    fieldnames.append("locus")
            rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unaligned-airr", type=Path, default=DEFAULT_UNALIGNED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--igblast-dir", type=Path, default=DEFAULT_IGBLAST)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--changeo-scripts", type=Path, default=DEFAULT_SCRIPTS)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--igblast-jobs", type=int, default=0)
    parser.add_argument("--igblast-threads", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--include-inferred-ogrdb", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = out_dir / "references"
    paths = build_references(ref_dir, out_dir, include_inferred=args.include_inferred_ogrdb)

    query_fasta = out_dir / "bcr_airr_queries.fasta"
    metadata = export_query_fasta(args.unaligned_airr, query_fasta)
    print(f"Exported {len(metadata)} query sequences to {query_fasta}")

    igdata = prepare_igdata(args.igblast_dir, out_dir)
    makeblastdb = args.igblast_dir / "bin" / "makeblastdb.exe"
    db_dir = igdata / "database"
    make_blast_db(makeblastdb, paths["v"], db_dir / "imgt_human_ig_v")
    make_blast_db(makeblastdb, paths["d"], db_dir / "imgt_human_ig_d")
    make_blast_db(makeblastdb, paths["j"], db_dir / "imgt_human_ig_j")
    make_blast_db(makeblastdb, paths["c"], db_dir / "imgt_human_ig_c")
    make_blast_db(makeblastdb, paths["dummy_d"], db_dir / "imgt_human_dummy_d")
    for key in ("igh_v", "igh_d", "igh_j", "igk_v", "igk_j", "igl_v", "igl_j"):
        make_blast_db(makeblastdb, paths[key], db_dir / f"imgt_human_{key}")
    ref_index = build_reference_index(paths)

    igblastn = args.igblast_dir / "bin" / "igblastn.exe"
    make_db = args.changeo_scripts / "MakeDb.py"
    create_germlines = args.changeo_scripts / "CreateGermlines.py"

    chunk_dir = out_dir / "chunks"
    chunk_specs = split_fasta_by_locus(query_fasta, chunk_dir, args.chunk_size, metadata)
    chunk_outs = [chunk_dir / f"{chunk.stem}_igblast.fmt7" for _, chunk in chunk_specs]
    if not args.resume_existing:
        for chunk_out in chunk_outs:
            chunk_out.unlink(missing_ok=True)
        (out_dir / "bcr_airr_ogrdb_igblast.fmt7").unlink(missing_ok=True)
    jobs = args.igblast_jobs if args.igblast_jobs > 0 else max(1, min(args.nproc, len(chunk_specs)))
    print(
        f"Running IgBLAST in {len(chunk_specs)} locus-specific chunks with {jobs} concurrent jobs "
        f"and {args.igblast_threads} thread(s) per job.",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        future_map = {
            pool.submit(
                run_igblast_chunk,
                igblastn,
                locus,
                chunk,
                out_file,
                out_dir,
                args.igblast_threads,
                metadata,
                ref_index,
                args.resume_existing,
            ): out_file
            for (locus, chunk), out_file in zip(chunk_specs, chunk_outs)
        }
        for future in as_completed(future_map):
            future.result()
            completed += 1
            if completed == 1 or completed % 5 == 0 or completed == len(chunk_outs):
                print(f"IgBLAST chunks complete: {completed}/{len(chunk_outs)}", flush=True)

    fmt7 = out_dir / "bcr_airr_ogrdb_igblast.fmt7"
    concatenate_text(chunk_outs, fmt7)

    run([
        str(args.python),
        str(make_db),
        "igblast",
        "-i",
        str(fmt7),
        "-s",
        str(query_fasta),
        "-r",
        str(paths["germline"]),
        "--asis-id",
        "--extended",
        "--partial",
        "--infer-junction",
        "--failed",
        "--format",
        "airr",
        "--outdir",
        str(out_dir),
        "--outname",
        "bcr_airr_changeo",
        "--nproc",
        str(args.nproc),
    ])
    db_pass = find_one(out_dir, "bcr_airr_changeo*db-pass.tsv")

    run([
        str(args.python),
        str(create_germlines),
        "-d",
        str(db_pass),
        "-r",
        str(paths["germline"]),
        "-g",
        "dmask",
        "full",
        "vonly",
        "--format",
        "airr",
        "--outdir",
        str(out_dir),
        "--outname",
        "bcr_airr_germlines",
        "--failed",
    ])
    germ_pass = find_one(out_dir, "bcr_airr_germlines*germ-pass.tsv")

    merged = out_dir / "bcr_changeo_germline_airr_with_metadata.tsv"
    merged_rows = merge_metadata(germ_pass, metadata, merged)
    summary = {
        "query_sequences": len(metadata),
        "igblast_fmt7": str(fmt7),
        "changeo_db_pass": str(db_pass),
        "germline_pass": str(germ_pass),
        "metadata_merged_airr": str(merged),
        "metadata_merged_rows": merged_rows,
    }
    (out_dir / "alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
