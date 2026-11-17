"""Application services: map versions, batches, runs, publication, export.

Consistency model
-----------------
* Every run is pinned to one ``map_versions`` row and reads that immutable
  snapshot.  Compartment geometry is never updated in place.
* Exactly one map version is active at a time (partial unique index).
* A run may publish only if, inside a ``BEGIN IMMEDIATE`` transaction:
    - it is still allowed to publish (not failed/cancelled),
    - its map version is still the active one,
    - no newer run has already published for the same batch.
  A late-finishing older computation therefore cannot overwrite a newer one,
  and a computation against a replaced map can never be published.
* Failed or superseded runs never touch ``batches.published_run_id``, so the
  previous downloadable detail survives.
* Downloads resolve ``published_run_id`` once and then read only that run's
  result rows, so a concurrent republication cannot produce a mixed file.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import db
from .attribution import (
    CONFLICT,
    MATCHED,
    REASON_LABELS,
    STATUS_LABELS,
    UNMATCHED,
    Attribution,
    Attributor,
    CompartmentGeometry,
    HotspotPoint,
)
from .parsing import (
    Compartment,
    InputError,
    Hotspot,
    geometry_to_wkb,
    parse_compartment_geojson,
    parse_hotspots_csv,
    sha256_bytes,
    wkb_to_geometry,
)

log = logging.getLogger("forest.service")

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SUPERSEDED = "superseded"

# Internal outcomes of a computation attempt.
PUBLISHED = "published"
SUPERSEDED_OLD_MAP = "superseded_old_map"
SUPERSEDED_NEWER = "superseded_newer"
ABORTED = "aborted"

RUN_STATUS_LABELS = {
    STATUS_QUEUED: "排队中",
    STATUS_RUNNING: "计算中",
    STATUS_SUCCEEDED: "已完成",
    STATUS_FAILED: "失败",
    STATUS_SUPERSEDED: "已作废",
}

# Post-restore verification outcomes (``verifications.status``).
VERIFY_PASSED = "passed"
VERIFY_FAILED = "failed"

# Machine-readable integrity problem kinds.  A verified detail is expected to
# contain exactly one result row per hotspot source row, and its row counts
# must match the published summary counters.
ISSUE_MISSING_ROWS = "missing_rows"
ISSUE_EXTRA_ROWS = "extra_rows"
ISSUE_DUPLICATE_ROWS = "duplicate_rows"
ISSUE_TOTAL_MISMATCH = "total_mismatch"
ISSUE_STATUS_MISMATCH = "status_mismatch"
ISSUE_UNKNOWN_STATUS = "unknown_status"
ISSUE_RUN_NOT_PUBLISHED = "run_not_succeeded"
ISSUE_MAP_MISSING = "map_missing"

ISSUE_LABELS = {
    ISSUE_MISSING_ROWS: "缺行：部分来源热点没有归属结果",
    ISSUE_EXTRA_ROWS: "额外行：结果含来源之外的热点序号",
    ISSUE_DUPLICATE_ROWS: "重复行：同一来源热点存在多条结果",
    ISSUE_TOTAL_MISMATCH: "总数不符：结果行数与发布统计的来源热点数不一致",
    ISSUE_STATUS_MISMATCH: "状态汇总不符：实际行数与发布统计（已赋码/冲突/未匹配）不一致",
    ISSUE_UNKNOWN_STATUS: "状态值异常：存在未登记的归属状态",
    ISSUE_RUN_NOT_PUBLISHED: "计算记录不是已发布成功状态",
    ISSUE_MAP_MISSING: "计算绑定的林班底图版本缺失",
}

VERIFY_ISSUE_LIMIT = 200

RESULT_COLUMNS = [
    "来源文件",
    "来源行号",
    "热点号",
    "经度",
    "纬度",
    "UTC时刻",
    "归属状态",
    "原因类型",
    "原因说明",
    "林班号",
    "候选林班号",
    "林班底图版本",
    "计算记录号",
]


class ServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class MapInfo:
    id: int
    filename: str
    sha256: str
    feature_count: int
    repaired_count: int
    created_at: str
    is_active: bool


def _row_to_map(row: sqlite3.Row) -> MapInfo:
    return MapInfo(
        id=row["id"],
        filename=row["filename"],
        sha256=row["sha256"],
        feature_count=row["feature_count"],
        repaired_count=row["repaired_count"],
        created_at=row["created_at"],
        is_active=bool(row["is_active"]),
    )


def get_active_map(conn: sqlite3.Connection) -> MapInfo | None:
    row = conn.execute("SELECT * FROM map_versions WHERE is_active = 1").fetchone()
    return _row_to_map(row) if row else None


def get_map(conn: sqlite3.Connection, map_id: int) -> MapInfo | None:
    row = conn.execute("SELECT * FROM map_versions WHERE id = ?", (map_id,)).fetchone()
    return _row_to_map(row) if row else None


def list_maps(conn: sqlite3.Connection, limit: int = 20) -> list[MapInfo]:
    rows = conn.execute(
        "SELECT * FROM map_versions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_map(row) for row in rows]


def publish_map(db_path: str, data: bytes, filename: str) -> tuple[MapInfo, list[int]]:
    """Parse and atomically activate a new compartment map version.

    Returns the new version and ids of batches whose previously published
    result was produced against an older version (they can be recomputed).
    """

    compartments, repaired = parse_compartment_geojson(data, filename)
    digest = sha256_bytes(data)
    created_at = db.utc_now()
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            conn.execute("UPDATE map_versions SET is_active = 0 WHERE is_active = 1")
            cur = conn.execute(
                """
                INSERT INTO map_versions
                    (filename, sha256, feature_count, repaired_count, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (filename, digest, len(compartments), repaired, created_at),
            )
            version_id = int(cur.lastrowid)
            conn.executemany(
                """
                INSERT INTO compartments
                    (map_version_id, seq, code, geom, minx, miny, maxx, maxy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        version_id,
                        item.seq,
                        item.code,
                        geometry_to_wkb(item.geometry),
                        item.minx,
                        item.miny,
                        item.maxx,
                        item.maxy,
                    )
                    for item in compartments
                ],
            )
            stale_rows = conn.execute(
                """
                SELECT DISTINCT b.id
                FROM batches b
                JOIN runs r ON r.id = b.published_run_id
                WHERE r.map_version_id != ?
                ORDER BY b.id
                """,
                (version_id,),
            ).fetchall()
            info = _row_to_map(
                conn.execute("SELECT * FROM map_versions WHERE id = ?", (version_id,)).fetchone()
            )
    return info, [int(row["id"]) for row in stale_rows]


def _insert_hotspots(conn: sqlite3.Connection, batch_id: int, hotspots: Sequence[Hotspot]) -> None:
    conn.executemany(
        """
        INSERT INTO hotspots (batch_id, seq, source_row, hotspot_no, lon, lat, observed_at_utc, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                batch_id,
                item.seq,
                item.source_row,
                item.hotspot_no,
                item.lon,
                item.lat,
                item.observed_at_utc,
                json.dumps(item.raw, ensure_ascii=False),
            )
            for item in hotspots
        ],
    )


def create_batch(db_path: str, data: bytes, filename: str) -> tuple[int, int | None]:
    """Store a hotspot batch and (if a map exists) enqueue its first run."""

    hotspots = parse_hotspots_csv(data, filename)
    digest = sha256_bytes(data)
    created_at = db.utc_now()
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            cur = conn.execute(
                """
                INSERT INTO batches (name, filename, sha256, row_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (Path(filename).stem or "热点批次", filename, digest, len(hotspots), created_at),
            )
            batch_id = int(cur.lastrowid)
            _insert_hotspots(conn, batch_id, hotspots)
            active = get_active_map(conn)
            run_id = None
            if active is not None:
                run_id = _insert_run(conn, batch_id, active.id)
    return batch_id, run_id


def enqueue_recompute(db_path: str, batch_id: int) -> int:
    """Queue a run against the currently active map version.

    If the newest run for this batch already uses the active version and is
    still queued/running, it is reused rather than duplicating the work.
    """

    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            batch = conn.execute("SELECT id FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if batch is None:
                raise ServiceError("热点批次不存在")
            active = get_active_map(conn)
            if active is None:
                raise ServiceError("尚未上传林班底图，无法计算")
            latest = conn.execute(
                "SELECT id, status, map_version_id FROM runs WHERE batch_id = ? ORDER BY id DESC LIMIT 1",
                (batch_id,),
            ).fetchone()
            if (
                latest is not None
                and latest["map_version_id"] == active.id
                and latest["status"] in (STATUS_QUEUED, STATUS_RUNNING)
            ):
                return int(latest["id"])
            return _insert_run(conn, batch_id, active.id)


def _insert_run(conn: sqlite3.Connection, batch_id: int, map_version_id: int) -> int:
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM hotspots WHERE batch_id = ?", (batch_id,)
    ).fetchone()["n"]
    cur = conn.execute(
        """
        INSERT INTO runs (batch_id, map_version_id, status, created_at, total_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        (batch_id, map_version_id, STATUS_QUEUED, db.utc_now(), total),
    )
    return int(cur.lastrowid)


def claim_next_run(db_path: str) -> tuple[int, str] | None:
    """Atomically take the oldest queued run; safe across app instances.

    Returns ``(run_id, attempt_token)``.  The token identifies this specific
    execution attempt: if a stalled worker is recovered and the run re-claimed
    elsewhere, the stale worker's late writes are rejected instead of
    corrupting the new attempt or marking it failed.
    """

    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            row = conn.execute(
                "SELECT id FROM runs WHERE status = ? ORDER BY id LIMIT 1", (STATUS_QUEUED,)
            ).fetchone()
            if row is None:
                return None
            run_id = int(row["id"])
            now = db.utc_now()
            token = uuid.uuid4().hex
            updated = conn.execute(
                """
                UPDATE runs SET status = ?, started_at = ?, heartbeat_at = ?, attempt_token = ?
                WHERE id = ? AND status = ?
                """,
                (STATUS_RUNNING, now, now, token, run_id, STATUS_QUEUED),
            ).rowcount
            if updated != 1:
                return None
            return run_id, token


def _load_run_inputs(
    db_path: str, run_id: int
) -> tuple[sqlite3.Row, list[HotspotPoint], list[CompartmentGeometry], MapInfo]:
    """Read one consistent snapshot of hotspots + the pinned map version."""

    with db.connection(db_path) as conn:
        with db.read_tx(conn):
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                raise ServiceError("计算记录不存在")
            batch = conn.execute(
                "SELECT * FROM batches WHERE id = ?", (run["batch_id"],)
            ).fetchone()
            map_info = get_map(conn, int(run["map_version_id"]))
            if map_info is None:
                raise ServiceError("计算所用的林班底图版本不存在")
            hotspot_rows = conn.execute(
                """
                SELECT seq, source_row, hotspot_no, lon, lat, observed_at_utc
                FROM hotspots WHERE batch_id = ? ORDER BY seq
                """,
                (run["batch_id"],),
            ).fetchall()
            compartments = [
                CompartmentGeometry(
                    seq=int(row["seq"]),
                    code=row["code"],
                    geometry=wkb_to_geometry(row["geom"]),
                )
                for row in conn.execute(
                    "SELECT seq, code, geom FROM compartments WHERE map_version_id = ? ORDER BY seq",
                    (map_info.id,),
                )
            ]
            hotspots = [HotspotPoint(*row) for row in hotspot_rows]
    return run, hotspots, compartments, map_info


def execute_run(db_path: str, run_id: int, attempt_token: str | None = None, chunk_size: int = 2000) -> bool:
    """Compute and publish a claimed run. Returns True if published.

    ``attempt_token`` is the token returned by :func:`claim_next_run`.  When it
    is supplied, every write is conditional on the run still belonging to this
    attempt, so a worker that was stalled and then recovered cannot delete or
    overwrite the results of the replacement attempt.  Passing ``None`` is only
    for tests and for callers that hold the run exclusively.
    """

    try:
        run, hotspots, compartments, map_info = _load_run_inputs(db_path, run_id)
        attributor = Attributor(compartments)
        counts, total, outcome = _compute_rows(
            db_path, run_id, attempt_token, hotspots, attributor, chunk_size
        )
        if outcome == PUBLISHED:
            return True
        _discard_results(db_path, run_id, attempt_token)
        if outcome == SUPERSEDED_OLD_MAP:
            if _mark_superseded(db_path, run_id, attempt_token):
                _requeue_against_active_map(db_path, int(run["batch_id"]))
        elif outcome == SUPERSEDED_NEWER:
            _mark_superseded(db_path, run_id, attempt_token)
        return False
    except Exception as exc:  # noqa: BLE001 - failure must be recorded, not raised
        log.exception("run %s failed", run_id)
        if _mark_failed(db_path, run_id, str(exc) or exc.__class__.__name__, attempt_token):
            _discard_results(db_path, run_id, attempt_token)
        return False


def _token_clause(attempt_token: str | None) -> tuple[str, tuple[Any, ...]]:
    """SQL fragment pinning a mutation to one attempt.

    ``None`` means the caller holds the run exclusively (tests, maintenance)
    and no attempt check is applied.
    """

    if attempt_token is None:
        return "", ()
    return " AND attempt_token = ?", (attempt_token,)


def _owns_attempt(
    conn: sqlite3.Connection, run_id: int, attempt_token: str | None, *, require_running: bool
) -> bool:
    row = conn.execute(
        "SELECT status, attempt_token FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return False
    if require_running and row["status"] != STATUS_RUNNING:
        return False
    if attempt_token is not None and row["attempt_token"] != attempt_token:
        return False
    return True


def _compute_rows(
    db_path: str,
    run_id: int,
    attempt_token: str | None,
    hotspots: Sequence[HotspotPoint],
    attributor: Attributor,
    chunk_size: int,
) -> tuple[dict[str, int], int, str]:
    counts = {MATCHED: 0, CONFLICT: 0, UNMATCHED: 0}
    total = len(hotspots)
    last_heartbeat = time.monotonic()
    for start in range(0, total, chunk_size):
        chunk = hotspots[start : start + chunk_size]
        rows: list[tuple[Any, ...]] = []
        for hotspot in chunk:
            attribution = attributor.attribute_one(hotspot)
            counts[attribution.status] += 1
            rows.append(
                (
                    run_id,
                    hotspot.seq,
                    attribution.status,
                    attribution.reason_kind,
                    attribution.reason,
                    db.dump_codes(list(attribution.codes)),
                )
            )
        with db.connection(db_path) as conn:
            with db.write_tx(conn):
                if not _owns_attempt(conn, run_id, attempt_token, require_running=True):
                    return counts, total, ABORTED
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO results
                        (run_id, hotspot_seq, status, reason_kind, reason, codes_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                if time.monotonic() - last_heartbeat > 5:
                    clause, params = _token_clause(attempt_token)
                    conn.execute(
                        f"UPDATE runs SET heartbeat_at = ? WHERE id = ?{clause}",
                        (db.utc_now(), run_id, *params),
                    )
                    last_heartbeat = time.monotonic()
    return counts, total, _publish_run(db_path, run_id, attempt_token, counts, total)


def _publish_run(
    db_path: str,
    run_id: int,
    attempt_token: str | None,
    counts: dict[str, int],
    total: int,
) -> str:
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None or not _owns_attempt(
                conn, run_id, attempt_token, require_running=True
            ):
                return ABORTED
            active = get_active_map(conn)
            if active is None or active.id != run["map_version_id"]:
                # Map was replaced while this run was computing. Publishing it
                # would break "one published detail = one map version".
                return SUPERSEDED_OLD_MAP
            batch = conn.execute(
                "SELECT published_run_id FROM batches WHERE id = ?", (run["batch_id"],)
            ).fetchone()
            if batch is not None and batch["published_run_id"] is not None:
                previous = conn.execute(
                    "SELECT id FROM runs WHERE id = ?", (batch["published_run_id"],)
                ).fetchone()
                if previous is not None and int(previous["id"]) > run_id:
                    # A newer computation already published; late old result loses.
                    return SUPERSEDED_NEWER
            now = db.utc_now()
            clause, params = _token_clause(attempt_token)
            success = conn.execute(
                f"""
                UPDATE runs SET status = ?, published = 1, finished_at = ?, error = NULL,
                    matched_count = ?, conflict_count = ?, unmatched_count = ?, total_count = ?
                WHERE id = ? AND status = ? AND published = 0{clause}
                """,
                (
                    STATUS_SUCCEEDED,
                    now,
                    counts[MATCHED],
                    counts[CONFLICT],
                    counts[UNMATCHED],
                    total,
                    run_id,
                    STATUS_RUNNING,
                    *params,
                ),
            ).rowcount
            if success != 1:
                return ABORTED
            conn.execute(
                "UPDATE batches SET published_run_id = ? WHERE id = ?", (run_id, run["batch_id"])
            )
            return PUBLISHED


def _requeue_against_active_map(db_path: str, batch_id: int) -> int | None:
    """After a map swap invalidated a computation, retry against the new map.

    Skips if a run for this batch is already queued/running on the active
    version, so two instances racing to requeue create only one run.
    """

    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            if conn.execute("SELECT 1 FROM batches WHERE id = ?", (batch_id,)).fetchone() is None:
                return None
            active = get_active_map(conn)
            if active is None:
                return None
            existing = conn.execute(
                """
                SELECT 1 FROM runs
                WHERE batch_id = ? AND map_version_id = ? AND status IN (?, ?)
                LIMIT 1
                """,
                (batch_id, active.id, STATUS_QUEUED, STATUS_RUNNING),
            ).fetchone()
            if existing is not None:
                return None
            return _insert_run(conn, batch_id, active.id)


def _mark_superseded(db_path: str, run_id: int, attempt_token: str | None = None) -> bool:
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            if not _owns_attempt(conn, run_id, attempt_token, require_running=True):
                return False
            clause, params = _token_clause(attempt_token)
            conn.execute(
                f"""
                UPDATE runs SET status = ?, finished_at = ?,
                    error = COALESCE(error, '林班底图已更换或已有更新计算发布，本次结果未发布')
                WHERE id = ? AND status = ?{clause}
                """,
                (STATUS_SUPERSEDED, db.utc_now(), run_id, STATUS_RUNNING, *params),
            )
            return True


def _mark_failed(
    db_path: str, run_id: int, message: str, attempt_token: str | None = None
) -> bool:
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            if not _owns_attempt(conn, run_id, attempt_token, require_running=False):
                return False
            # Only the attempt that owns the run may mark it failed; a stale
            # worker must not turn a re-claimed run into a failure.
            clause, params = _token_clause(attempt_token)
            updated = conn.execute(
                f"""
                UPDATE runs SET status = ?, finished_at = ?, error = ?
                WHERE id = ? AND status IN (?, ?){clause}
                """,
                (
                    STATUS_FAILED,
                    db.utc_now(),
                    message[:2000],
                    run_id,
                    STATUS_QUEUED,
                    STATUS_RUNNING,
                    *params,
                ),
            ).rowcount
            return updated == 1


def _discard_results(db_path: str, run_id: int, attempt_token: str | None = None) -> None:
    """Delete partial results for a run that did not publish.

    With a token, the delete only happens if this attempt still owns the run:
    a stale worker must not wipe rows written by the replacement attempt.
    """

    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            if not _owns_attempt(conn, run_id, attempt_token, require_running=False):
                return
            conn.execute("DELETE FROM results WHERE run_id = ?", (run_id,))


def recover_stale_runs(db_path: str, stale_seconds: int) -> int:
    """Requeue runs abandoned by a crashed instance.

    Only runs whose heartbeat is older than ``stale_seconds`` are touched, so a
    healthy long-running computation on another instance is never stolen.
    """

    cutoff = datetime.now(timezone.utc).timestamp() - stale_seconds
    recovered = 0
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            rows = conn.execute(
                "SELECT id, started_at, heartbeat_at FROM runs WHERE status = ?", (STATUS_RUNNING,)
            ).fetchall()
            for row in rows:
                stamp = row["heartbeat_at"] or row["started_at"]
                try:
                    when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
                except (AttributeError, ValueError):
                    when = 0
                if when < cutoff:
                    conn.execute(
                        """
                        UPDATE runs SET status = ?, started_at = NULL, heartbeat_at = NULL,
                            error = '上一次计算进程中断，已重新排队'
                        WHERE id = ?
                        """,
                        (STATUS_QUEUED, row["id"]),
                    )
                    conn.execute("DELETE FROM results WHERE run_id = ?", (row["id"],))
                    recovered += 1
    return recovered


# --------------------------------------------------------------------------
# Post-restore verification, supply blocking and rebuild
# --------------------------------------------------------------------------
#
# A verification reads one WAL snapshot of the *currently published* run and
# checks, per hotspot source row, that exactly one result row exists and that
# the stored status counters match the rows actually present.  Applying the
# outcome is a compare-and-swap inside a ``BEGIN IMMEDIATE`` transaction:
#
#   * the verification re-reads ``batches.published_run_id`` and only acts when
#     the same run is still published, so a verification whose snapshot was
#     taken before a republication can never block the new result;
#   * a failed verification inserts (or keeps) an *active block* for the batch
#     and enqueues a rebuild against the active map in the same transaction;
#   * a failed rebuild never touches the block, so the bad detail stays
#     un-downloadable; the block is recovered only by a *passed* verification
#     of the run that is published at that moment.
#
# Blocks therefore protect the batch (the supply channel) while each block row
# records the exact run id that was proven bad, which keeps an old verification
# from touching a newer result.

VALID_RESULT_STATUSES = (MATCHED, CONFLICT, UNMATCHED)


def _make_issue(kind: str, detail: str, seqs: Sequence[int] | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": ISSUE_LABELS.get(kind, kind),
        "detail": detail,
        "seqs": list(seqs or []),
    }


def _collect_verification_snapshot(
    conn: sqlite3.Connection, batch_id: int, run_id: int
) -> dict[str, Any]:
    """Read every fact a verification needs from one consistent snapshot."""

    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    map_exists = conn.execute(
        "SELECT 1 FROM map_versions WHERE id = ?", (run["map_version_id"],)
    ).fetchone() if run is not None else None
    hotspots = conn.execute(
        "SELECT seq, source_row, hotspot_no FROM hotspots WHERE batch_id = ? ORDER BY seq",
        (batch_id,),
    ).fetchall()
    seq_rows = conn.execute(
        "SELECT hotspot_seq, COUNT(*) AS n FROM results WHERE run_id = ? GROUP BY hotspot_seq",
        (run_id,),
    ).fetchall()
    status_rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM results WHERE run_id = ? GROUP BY status",
        (run_id,),
    ).fetchall()
    return {
        "run": run,
        "map_exists": map_exists is not None,
        "hotspots": hotspots,
        "seq_rows": seq_rows,
        "status_rows": status_rows,
    }


def _build_verification_issues(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    run = snapshot["run"]
    hotspots: Sequence[sqlite3.Row] = snapshot["hotspots"]
    hotspot_info = {int(row["seq"]): row for row in hotspots}
    total_sources = len(hotspots)

    if run is None or run["status"] != STATUS_SUCCEEDED or not run["published"]:
        issues.append(
            _make_issue(
                ISSUE_RUN_NOT_PUBLISHED,
                "受检计算记录不存在或不处于已发布成功状态，明细不可供数",
            )
        )
        return issues
    if not snapshot["map_exists"]:
        issues.append(
            _make_issue(
                ISSUE_MAP_MISSING,
                f"计算 run #{run['id']} 绑定的林班底图 v{run['map_version_id']} 在库中缺失",
            )
        )

    counts_by_seq = {int(row["hotspot_seq"]): int(row["n"]) for row in snapshot["seq_rows"]}
    result_seqs = set(counts_by_seq)
    source_seqs = set(hotspot_info)

    missing = sorted(source_seqs - result_seqs)
    extra = sorted(result_seqs - source_seqs)
    duplicates = sorted(seq for seq, n in counts_by_seq.items() if n > 1)

    if missing:
        sample = [
            f"seq {seq}（行 {hotspot_info[seq]['source_row']} · {hotspot_info[seq]['hotspot_no']}）"
            for seq in missing[:10]
        ]
        issues.append(
            _make_issue(
                ISSUE_MISSING_ROWS,
                f"缺 {len(missing)} 条结果，例如：{'、'.join(sample)}",
                missing,
            )
        )
    if extra:
        issues.append(
            _make_issue(
                ISSUE_EXTRA_ROWS,
                f"存在 {len(extra)} 个来源之外的热点序号：{', '.join(map(str, extra[:20]))}",
                extra,
            )
        )
    if duplicates:
        issues.append(
            _make_issue(
                ISSUE_DUPLICATE_ROWS,
                f"{len(duplicates)} 个来源热点对应多条结果：{', '.join(map(str, duplicates[:20]))}",
                duplicates,
            )
        )

    actual_total = sum(counts_by_seq.values())
    distinct_total = len(result_seqs)
    if actual_total != total_sources or distinct_total != total_sources:
        issues.append(
            _make_issue(
                ISSUE_TOTAL_MISMATCH,
                f"来源热点 {total_sources} 条；结果物理行 {actual_total} 行、覆盖来源 {distinct_total} 条；"
                f"发布统计 total_count={run['total_count']}",
            )
        )

    actual_by_status = {row["status"]: int(row["n"]) for row in snapshot["status_rows"]}
    unknown = sorted(status for status in actual_by_status if status not in VALID_RESULT_STATUSES)
    if unknown:
        issues.append(
            _make_issue(
                ISSUE_UNKNOWN_STATUS,
                f"出现未登记的归属状态：{', '.join(unknown)}",
            )
        )
    expected_by_status = {
        MATCHED: int(run["matched_count"]),
        CONFLICT: int(run["conflict_count"]),
        UNMATCHED: int(run["unmatched_count"]),
    }
    mismatches = []
    for status, expected in expected_by_status.items():
        actual = actual_by_status.get(status, 0)
        if actual != expected:
            mismatches.append(
                f"{STATUS_LABELS[status]} 实际 {actual} 行 vs 发布统计 {expected} 行"
            )
    if mismatches:
        issues.append(_make_issue(ISSUE_STATUS_MISMATCH, "；".join(mismatches)))

    return issues


def _enqueue_recompute_locked(conn: sqlite3.Connection, batch_id: int) -> int | None:
    """Queue an active-map rebuild; must run inside a write transaction.

    Reuses an already queued/running run on the active version so repeated
    failed verifications (or two racing instances) never pile up duplicate
    queue entries.
    """

    active = get_active_map(conn)
    if active is None:
        return None
    existing = conn.execute(
        """
        SELECT id FROM runs
        WHERE batch_id = ? AND map_version_id = ? AND status IN (?, ?)
        LIMIT 1
        """,
        (batch_id, active.id, STATUS_QUEUED, STATUS_RUNNING),
    ).fetchone()
    if existing is not None:
        return int(existing["id"])
    return _insert_run(conn, batch_id, active.id)


def _insert_active_block(
    conn: sqlite3.Connection, batch_id: int, run_id: int, reason: str, now: str
) -> int:
    """Insert an active block unless one already exists (serialised callers)."""

    cur = conn.execute(
        """
        INSERT INTO supply_blocks (batch_id, run_id, reason, created_at)
        SELECT ?, ?, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM supply_blocks WHERE batch_id = ? AND recovered_at IS NULL
        )
        """,
        (batch_id, run_id, reason, now, batch_id),
    )
    if cur.rowcount:
        return int(cur.lastrowid)
    row = conn.execute(
        "SELECT id FROM supply_blocks WHERE batch_id = ? AND recovered_at IS NULL",
        (batch_id,),
    ).fetchone()
    return int(row["id"])


def verify_published_batch(
    db_path: str,
    batch_id: int,
    pre_commit: Any | None = None,
) -> dict[str, Any]:
    """Verify the currently published detail of one batch.

    Returns a summary dict (also persisted in ``verifications``).  When the
    published run is replaced between the snapshot read and the commit, the
    verification is recorded as not applied and changes no block state.

    ``pre_commit(conn, context)`` is an optional test seam invoked inside the
    write transaction before the outcome is committed, to simulate a
    republication interleaving between two instances.
    """

    with db.connection(db_path) as conn:
        with db.read_tx(conn):
            batch = conn.execute(
                "SELECT * FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise ServiceError("热点批次不存在")
            run_id = batch["published_run_id"]
            if run_id is None:
                raise ServiceError("该批次还没有已发布的计算结果，无需核验")
            snapshot = _collect_verification_snapshot(conn, batch_id, int(run_id))
        issues = _build_verification_issues(snapshot)
        passed = not issues
        run = snapshot["run"]
        map_version_id = int(run["map_version_id"]) if run is not None else None
        now = db.utc_now()

        with db.write_tx(conn):
            if pre_commit is not None:
                # Test seam: runs inside the exclusive write transaction but
                # before the compare-and-swap, modelling work interleaved by
                # another instance between the snapshot read and the commit.
                pre_commit(
                    conn,
                    {
                        "batch_id": batch_id,
                        "run_id": int(run_id),
                        "snapshot_passed": passed,
                    },
                )
            current = conn.execute(
                "SELECT published_run_id FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if current is None or current["published_run_id"] != run_id:
                # The result was republished after our snapshot: this old
                # verification must not block or recover the new result.
                cur = conn.execute(
                    """
                    INSERT INTO verifications
                        (batch_id, run_id, map_version_id, status, applied,
                         issue_count, issues_json, note, checked_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        run_id,
                        map_version_id,
                        VERIFY_PASSED if passed else VERIFY_FAILED,
                        len(issues),
                        json.dumps(issues[:VERIFY_ISSUE_LIMIT], ensure_ascii=False),
                        "核验期间该批次已重新发布，本次结果未应用",
                        now,
                    ),
                )
                return {
                    "verification_id": int(cur.lastrowid),
                    "batch_id": batch_id,
                    "run_id": int(run_id),
                    "map_version_id": map_version_id,
                    "status": VERIFY_PASSED if passed else VERIFY_FAILED,
                    "applied": False,
                    "issues": issues,
                    "issue_count": len(issues),
                    "blocked": False,
                    "recovered": False,
                    "requeued_run_id": None,
                }

            blocked = recovered = False
            requeued_run_id: int | None = None
            verification_id: int | None = None
            reason = "；".join(issue["label"] for issue in issues[:5]) or "核验发现明细完整性问题"

            if passed:
                recovered_row = conn.execute(
                    "SELECT id FROM supply_blocks WHERE batch_id = ? AND recovered_at IS NULL",
                    (batch_id,),
                ).fetchone()
                if recovered_row is not None:
                    # Insert the verification first so the recovery can point
                    # at the exact check that restored supply.
                    verification_id = _insert_verification(
                        conn, batch_id, int(run_id), map_version_id, True, issues, now,
                        note="核验合格，恢复下载供数",
                    )
                    conn.execute(
                        """
                        UPDATE supply_blocks
                        SET recovered_at = ?, recovered_verification_id = ?
                        WHERE id = ? AND recovered_at IS NULL
                        """,
                        (now, verification_id, int(recovered_row["id"])),
                    )
                    recovered = True
                else:
                    verification_id = _insert_verification(
                        conn, batch_id, int(run_id), map_version_id, True, issues, now,
                        note="核验合格，明细可继续供数",
                    )
            else:
                _insert_active_block(conn, batch_id, int(run_id), reason, now)
                blocked = True
                requeued_run_id = _enqueue_recompute_locked(conn, batch_id)
                note = "核验不合格，已封禁后续下载并按当前底图排队重建"
                if requeued_run_id is None:
                    note += "（当前无活动底图，未能排队）"
                verification_id = _insert_verification(
                    conn, batch_id, int(run_id), map_version_id, False, issues, now, note=note
                )

            return {
                "verification_id": verification_id,
                "batch_id": batch_id,
                "run_id": int(run_id),
                "map_version_id": map_version_id,
                "status": VERIFY_PASSED if passed else VERIFY_FAILED,
                "applied": True,
                "issues": issues,
                "issue_count": len(issues),
                "blocked": blocked,
                "recovered": recovered,
                "requeued_run_id": requeued_run_id,
            }


def _insert_verification(
    conn: sqlite3.Connection,
    batch_id: int,
    run_id: int,
    map_version_id: int | None,
    passed: bool,
    issues: Sequence[dict[str, Any]],
    now: str,
    note: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO verifications
            (batch_id, run_id, map_version_id, status, applied,
             issue_count, issues_json, note, checked_at)
        VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
        """,
        (
            batch_id,
            run_id,
            map_version_id,
            VERIFY_PASSED if passed else VERIFY_FAILED,
            len(issues),
            json.dumps(list(issues)[:VERIFY_ISSUE_LIMIT], ensure_ascii=False),
            note,
            now,
        ),
    )
    return int(cur.lastrowid)


def verify_all_published_batches(db_path: str) -> list[dict[str, Any]]:
    """Verify every batch that currently has a published detail."""

    with db.connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM batches WHERE published_run_id IS NOT NULL ORDER BY id"
        ).fetchall()
    summaries = []
    for row in rows:
        try:
            summaries.append(verify_published_batch(db_path, int(row["id"])))
        except ServiceError as exc:  # pragma: no cover - defensive only
            summaries.append({"batch_id": int(row["id"]), "applied": False, "error": str(exc)})
    return summaries


def get_active_block(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM supply_blocks WHERE batch_id = ? AND recovered_at IS NULL",
        (batch_id,),
    ).fetchone()
    return dict(row) if row else None


def is_batch_blocked(conn: sqlite3.Connection, batch_id: int) -> bool:
    return get_active_block(conn, batch_id) is not None


def list_verifications(conn: sqlite3.Connection, batch_id: int, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM verifications WHERE batch_id = ? ORDER BY id DESC LIMIT ?",
        (batch_id, limit),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["issues"] = json.loads(item.pop("issues_json"))
        except (TypeError, ValueError):
            item["issues"] = []
        item["status_label"] = "合格" if item["status"] == VERIFY_PASSED else "不合格"
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Queries for the web UI
# --------------------------------------------------------------------------


def list_batches(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT b.*,
               pr.id AS pub_run_id, pr.map_version_id AS pub_map_id, pr.status AS pub_status,
               pr.total_count AS pub_total, pr.matched_count AS pub_matched,
               pr.conflict_count AS pub_conflict, pr.unmatched_count AS pub_unmatched,
               pm.filename AS pub_map_filename,
               lr.id AS latest_run_id, lr.status AS latest_status,
               lr.map_version_id AS latest_map_id,
               (SELECT COUNT(1) FROM supply_blocks sb
                  WHERE sb.batch_id = b.id AND sb.recovered_at IS NULL) AS blocked,
               (SELECT MAX(id) FROM verifications v WHERE v.batch_id = b.id) AS last_verify_id
        FROM batches b
        LEFT JOIN runs pr ON pr.id = b.published_run_id
        LEFT JOIN map_versions pm ON pm.id = pr.map_version_id
        LEFT JOIN runs lr ON lr.id = (
            SELECT id FROM runs WHERE batch_id = b.id ORDER BY id DESC LIMIT 1
        )
        ORDER BY b.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_batch(conn: sqlite3.Connection, batch_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return dict(row) if row else None


def list_runs(conn: sqlite3.Connection, batch_id: int, limit: int = 30) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.*, m.filename AS map_filename, m.sha256 AS map_sha256
        FROM runs r
        LEFT JOIN map_versions m ON m.id = r.map_version_id
        WHERE r.batch_id = ?
        ORDER BY r.id DESC
        LIMIT ?
        """,
        (batch_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT r.*, m.filename AS map_filename, m.sha256 AS map_sha256
        FROM runs r LEFT JOIN map_versions m ON m.id = r.map_version_id
        WHERE r.id = ?
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def result_counts(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT status, reason_kind, COUNT(*) AS n
        FROM results WHERE run_id = ?
        GROUP BY status, reason_kind
        ORDER BY status, n DESC
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "status": row["status"],
            "status_label": STATUS_LABELS.get(row["status"], row["status"]),
            "reason_kind": row["reason_kind"],
            "reason_label": REASON_LABELS.get(row["reason_kind"], row["reason_kind"]),
            "count": row["n"],
        }
        for row in rows
    ]


def preview_results(
    conn: sqlite3.Connection, run_id: int, limit: int = 200, only_problems: bool = False
) -> list[dict[str, Any]]:
    clause = "AND r.status != 'matched'" if only_problems else ""
    rows = conn.execute(
        f"""
        SELECT h.source_row, h.hotspot_no, h.lon, h.lat, h.observed_at_utc,
               r.status, r.reason_kind, r.reason, r.codes_json
        FROM results r JOIN hotspots h
          ON h.batch_id = (SELECT batch_id FROM runs WHERE id = ?) AND h.seq = r.hotspot_seq
        WHERE r.run_id = ? {clause}
        ORDER BY h.seq
        LIMIT ?
        """,
        (run_id, run_id, limit),
    ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        codes = db.load_codes(item.pop("codes_json"))
        item["code"] = codes[0] if item["status"] == MATCHED and codes else ""
        item["codes"] = codes
        item["status_label"] = STATUS_LABELS.get(item["status"], item["status"])
        item["reason_label"] = REASON_LABELS.get(item["reason_kind"], item["reason_kind"])
        out.append(item)
    return out


def get_published_run(conn: sqlite3.Connection, batch: dict[str, Any]) -> dict[str, Any] | None:
    run_id = batch.get("published_run_id")
    return get_run(conn, int(run_id)) if run_id is not None else None


def is_run_active(conn: sqlite3.Connection, batch_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM runs WHERE batch_id = ? AND status IN (?, ?) LIMIT 1",
        (batch_id, STATUS_QUEUED, STATUS_RUNNING),
    ).fetchone()
    return row is not None


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def iter_result_csv(db_path: str, batch_id: int, chunk: int = 1000) -> Iterator[str]:
    """Yield the published detail as CSV, pinned to one published run."""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")

    def drain() -> str:
        value = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)
        return value

    with db.connection(db_path) as conn:
        with db.read_tx(conn):
            batch = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
            if batch is None:
                raise ServiceError("热点批次不存在")
            block = get_active_block(conn, batch_id)
            if block is not None:
                raise ServiceError(
                    f"该批次明细核验不合格（run #{block['run_id']}），已封禁下载，"
                    "重建并重新核验合格后恢复供数"
                )
            run_id = batch["published_run_id"]
            if run_id is None:
                raise ServiceError("该批次还没有可下载的计算结果")
            run = conn.execute(
                """
                SELECT r.*, m.filename AS map_filename, m.id AS map_id
                FROM runs r JOIN map_versions m ON m.id = r.map_version_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise ServiceError("已发布的计算结果不存在")
            writer.writerow(RESULT_COLUMNS)
            yield "﻿" + drain()
            cursor = conn.execute(
                """
                SELECT h.source_row, h.hotspot_no, h.lon, h.lat, h.observed_at_utc,
                       r.status, r.reason_kind, r.reason, r.codes_json,
                       b.filename AS batch_filename, b.id AS batch_id,
                       m.filename AS map_filename, m.id AS map_id, rr.id AS run_id
                FROM results r
                JOIN runs rr ON rr.id = r.run_id
                JOIN hotspots h ON h.batch_id = rr.batch_id AND h.seq = r.hotspot_seq
                JOIN batches b ON b.id = rr.batch_id
                JOIN map_versions m ON m.id = rr.map_version_id
                WHERE r.run_id = ?
                ORDER BY h.seq
                """,
                (run_id,),
            )
            while True:
                rows = cursor.fetchmany(chunk)
                if not rows:
                    break
                for row in rows:
                    codes = db.load_codes(row["codes_json"])
                    matched = row["status"] == MATCHED and codes
                    writer.writerow(
                        [
                            row["batch_filename"],
                            row["source_row"],
                            row["hotspot_no"],
                            f"{row['lon']:.8f}".rstrip("0").rstrip("."),
                            f"{row['lat']:.8f}".rstrip("0").rstrip("."),
                            row["observed_at_utc"],
                            STATUS_LABELS.get(row["status"], row["status"]),
                            REASON_LABELS.get(row["reason_kind"], row["reason_kind"]),
                            row["reason"],
                            codes[0] if matched else "",
                            ";".join(codes) if not matched else "",
                            f"v{row['map_id']} · {row['map_filename']}",
                            f"run-{row['run_id']}",
                        ]
                    )
                yield drain()
            if buffer.tell():
                yield drain()
