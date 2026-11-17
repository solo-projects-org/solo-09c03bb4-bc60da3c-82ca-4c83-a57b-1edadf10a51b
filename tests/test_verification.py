"""Tests for post-restore verification, supply blocking and rebuild.

Coverage of the required interleaving guarantee:

* an old verification whose snapshot predates a republication is recorded as
  "not applied" and can never block the new result (``pre_commit`` seam and a
  real two-thread race);
* a still-published bad result is never left unblocked: the blocking
  compare-and-swap is re-checked inside the same ``BEGIN IMMEDIATE``
  transaction that enqueues the rebuild;
* failed rebuilds keep the block; only a passed verification of the currently
  published run restores downloads;
* healthy batches keep their normal download behaviour.
"""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.main import create_app


def make_client(tmp_path, start_worker=False):
    app = create_app(db_path=str(tmp_path / "forest.db"), start_worker=start_worker)
    return TestClient(app), app


def geojson(polygons):
    features = []
    for code, (minx, miny, maxx, maxy) in polygons.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"code": code},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [minx, miny],
                            [maxx, miny],
                            [maxx, maxy],
                            [minx, maxy],
                            [minx, miny],
                        ]
                    ],
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def csv_bytes(rows):
    lines = ["hotspot_no,lon,lat,observed_at"]
    lines += [",".join(map(str, row)) for row in rows]
    return ("\n".join(lines) + "\n").encode()


def drain(app):
    while True:
        claimed = service.claim_next_run(app.state.settings["db_path"])
        if claimed is None:
            return
        run_id, attempt_token = claimed
        service.execute_run(app.state.settings["db_path"], run_id, attempt_token=attempt_token)


def publish_one_batch(tmp_path, polygons=None, rows=None):
    polygons = polygons or {"A": (0, 0, 1, 1), "B": (1, 0, 2, 1)}
    rows = rows or [
        ("H1", 0.5, 0.5, "2026-01-01T00:00:00Z"),
        ("H2", 1, 0.5, "2026-01-01T00:01:00Z"),
        ("H3", 9, 9, "2026-01-01T00:02:00Z"),
    ]
    client, app = make_client(tmp_path)
    with client:
        client.post(
            "/maps",
            files={"file": ("m.geojson", geojson(polygons), "application/geo+json")},
        )
        client.post(
            "/batches",
            files={"file": ("h.csv", csv_bytes(rows), "text/csv")},
        )
        drain(app)
    return client, app


def test_healthy_published_result_passes_verification(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        r = client.post("/batches/1/verify", follow_redirects=False)
        assert r.status_code == 303
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is False
        assert payload["verifications"][0]["status"] == "passed"
        assert payload["verifications"][0]["issue_count"] == 0
        # Normal batch keeps its normal download behaviour.
        assert client.get("/batches/1/download.csv").status_code == 200

    summary = service.verify_published_batch(db_path, 1)
    assert summary["status"] == "passed" and summary["blocked"] is False
    with db.connection(db_path) as c:
        assert service.is_batch_blocked(c, 1) is False


def test_missing_result_rows_blocks_download_and_queues_rebuild(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        # Simulate backup corruption: two result rows disappear.
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute(
                    "DELETE FROM results WHERE run_id = ? AND hotspot_seq IN (1, 2)",
                    (published_run,),
                )

        r = client.post("/batches/1/verify", follow_redirects=False)
        assert r.status_code == 303
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        verify = payload["verifications"][0]
        assert verify["status"] == "failed"
        kinds = {issue["kind"] for issue in verify["issues"]}
        assert service.ISSUE_MISSING_ROWS in kinds
        assert service.ISSUE_TOTAL_MISMATCH in kinds
        missing = next(i for i in verify["issues"] if i["kind"] == service.ISSUE_MISSING_ROWS)
        assert set(missing["seqs"]) == {1, 2}

        # The bad result is blocked at both the route and the export layer.
        blocked_resp = client.get("/batches/1/download.csv")
        assert blocked_resp.status_code == 409
        with pytest.raises(service.ServiceError):
            next(service.iter_result_csv(db_path, 1))

        # A rebuild against the active map was enqueued in the same transaction.
        with db.connection(db_path) as c:
            queued = c.execute(
                "SELECT id, map_version_id, status FROM runs WHERE batch_id = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            block = c.execute("SELECT * FROM supply_blocks WHERE batch_id = 1").fetchone()
        assert queued["status"] == "queued"
        assert queued["map_version_id"] == 1
        assert block["recovered_at"] is None and block["run_id"] == published_run

        # Rebuild succeeds and publishes; the block still holds until re-checked.
        drain(app)
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        assert payload["published_run_id"] == queued["id"]
        assert client.get("/batches/1/download.csv").status_code == 409

        # Only a passed verification of the new result restores supply.
        r = client.post("/batches/1/verify", follow_redirects=False)
        assert r.status_code == 303
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is False
        assert payload["verifications"][0]["status"] == "passed"
        assert client.get("/batches/1/download.csv").status_code == 200
        with db.connection(db_path) as c:
            block = c.execute("SELECT * FROM supply_blocks WHERE batch_id = 1").fetchone()
            assert block["recovered_at"] is not None
            assert block["recovered_verification_id"] == payload["verifications"][0]["id"]


def test_extra_result_rows_are_detected(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute(
                    "INSERT INTO results (run_id, hotspot_seq, status, reason_kind, reason, codes_json) "
                    "VALUES (?, 99, 'matched', 'matched_interior', 'x', '[]')",
                    (published_run,),
                )
        client.post("/batches/1/verify", follow_redirects=False)
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        kinds = {i["kind"] for i in payload["verifications"][0]["issues"]}
        assert service.ISSUE_EXTRA_ROWS in kinds
        extra = next(i for i in payload["verifications"][0]["issues"] if i["kind"] == service.ISSUE_EXTRA_ROWS)
        assert extra["seqs"] == [99]


def test_duplicate_result_rows_are_detected(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        # The production schema's PK (run_id, hotspot_seq) prevents ordinary
        # duplicates; a partial restore can still leave a table without it, so
        # rebuild the table without the constraint and duplicate one source.
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute("ALTER TABLE results RENAME TO results_old")
                c.execute(
                    """
                    CREATE TABLE results (
                        run_id INTEGER NOT NULL, hotspot_seq INTEGER NOT NULL,
                        status TEXT NOT NULL, reason_kind TEXT NOT NULL,
                        reason TEXT NOT NULL, codes_json TEXT NOT NULL
                    )
                    """
                )
                c.execute("INSERT INTO results SELECT * FROM results_old")
                c.execute(
                    "INSERT INTO results SELECT * FROM results_old WHERE run_id = ? AND hotspot_seq = 0",
                    (published_run,),
                )
                c.execute("DROP TABLE results_old")
        client.post("/batches/1/verify", follow_redirects=False)
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        kinds = {i["kind"] for i in payload["verifications"][0]["issues"]}
        assert service.ISSUE_DUPLICATE_ROWS in kinds
        dup = next(i for i in payload["verifications"][0]["issues"] if i["kind"] == service.ISSUE_DUPLICATE_ROWS)
        assert dup["seqs"] == [0]


def test_published_statistics_mismatch_blocks(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                # Row set stays intact (one row per source), only the stored
                # summary counter is wrong after the restore.
                c.execute(
                    "UPDATE runs SET matched_count = matched_count + 1 WHERE id = ?",
                    (published_run,),
                )
        client.post("/batches/1/verify", follow_redirects=False)
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        kinds = {i["kind"] for i in payload["verifications"][0]["issues"]}
        assert service.ISSUE_STATUS_MISMATCH in kinds
        assert service.ISSUE_TOTAL_MISMATCH not in kinds


def test_failed_rebuild_keeps_block_until_passed_verification(tmp_path, monkeypatch):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute(
                    "DELETE FROM results WHERE run_id = ? AND hotspot_seq = 0",
                    (published_run,),
                )
        client.post("/batches/1/verify", follow_redirects=False)
        assert client.get("/api/batches/1").json()["blocked"] is True

        # First rebuild attempt fails; the block must remain in force.
        import app.attribution as attribution_module

        def boom(self, hotspot):
            raise RuntimeError("模拟重建故障")

        monkeypatch.setattr(attribution_module.Attributor, "attribute_one", boom)
        drain(app)
        monkeypatch.undo()
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is True
        assert payload["runs"][0]["status"] == "failed"
        assert client.get("/batches/1/download.csv").status_code == 409

        # Manual recompute + successful rebuild; still blocked until verified.
        client.post("/batches/1/recompute", follow_redirects=False)
        drain(app)
        assert client.get("/api/batches/1").json()["blocked"] is True
        client.post("/batches/1/verify", follow_redirects=False)
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is False
        assert client.get("/batches/1/download.csv").status_code == 200


def test_repeated_failed_verification_keeps_single_active_block(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute("DELETE FROM results WHERE run_id = ? AND hotspot_seq = 2", (published_run,))
        for _ in range(3):
            client.post("/batches/1/verify", follow_redirects=False)
        with db.connection(db_path) as c:
            active = c.execute(
                "SELECT COUNT(*) AS n FROM supply_blocks WHERE batch_id = 1 AND recovered_at IS NULL"
            ).fetchone()["n"]
            total = c.execute("SELECT COUNT(*) AS n FROM supply_blocks WHERE batch_id = 1").fetchone()["n"]
            queued = c.execute(
                "SELECT COUNT(*) AS n FROM runs WHERE batch_id = 1 AND status = 'queued'"
            ).fetchone()["n"]
            verifications = c.execute(
                "SELECT COUNT(*) AS n FROM verifications WHERE batch_id = 1"
            ).fetchone()["n"]
        assert active == 1 and total == 1
        assert queued == 1  # rebuild queue is deduped
        assert verifications == 3


def test_stale_verification_after_republication_is_not_applied(tmp_path):
    """Old verification must not block a result that replaced its snapshot."""

    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        old_run = client.get("/api/batches/1").json()["published_run_id"]

        def republish_before_commit(conn, context):
            # Runs inside the exclusive write transaction, before the
            # compare-and-swap re-read: emulates another instance's
            # republication landing between the snapshot read and this commit.
            if context["run_id"] != old_run:
                return
            cur = conn.execute(
                "INSERT INTO runs (batch_id, map_version_id, status, created_at, published, total_count) "
                "VALUES (1, 1, 'succeeded', ?, 1, 3)",
                (db.utc_now(),),
            )
            new_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO results "
                "SELECT ?, hotspot_seq, status, reason_kind, reason, codes_json FROM results WHERE run_id = ?",
                (new_id, old_run),
            )
            conn.execute("UPDATE batches SET published_run_id = ? WHERE id = 1", (new_id,))

        summary = service.verify_published_batch(db_path, 1, pre_commit=republish_before_commit)
        assert summary["applied"] is False
        assert summary["blocked"] is False
        payload = client.get("/api/batches/1").json()
        assert payload["blocked"] is False
        assert payload["verifications"][0]["applied"] is False
        # The new result remains downloadable: old verification did not block it.
        assert client.get("/batches/1/download.csv").status_code == 200


def test_verification_race_with_republication(tmp_path):
    """Real two-thread race: verification vs. a concurrent republication."""

    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        old_run = client.get("/api/batches/1").json()["published_run_id"]
        # Corrupt the old result: if the verification wins, it blocks run #1
        # and queues a rebuild; if publication wins, the verification goes
        # stale and never blocks the new run.  Either way, no bad active state
        # and no wrongful block of the new result may remain.
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute("DELETE FROM results WHERE run_id = ? AND hotspot_seq = 0", (old_run,))

        outcomes = []

        def do_verify():
            outcomes.append(("verify", service.verify_published_batch(db_path, 1)))

        def do_republish():
            client.post("/batches/1/recompute", follow_redirects=False)
            drain(app)
            outcomes.append(("republish", client.get("/api/batches/1").json()["published_run_id"]))

        t1 = threading.Thread(target=do_verify)
        t2 = threading.Thread(target=do_republish)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        payload = client.get("/api/batches/1").json()
        verify_outcome = next(value for name, value in outcomes if name == "verify")
        new_published = payload["published_run_id"]

        if not verify_outcome["applied"]:
            # Publication happened first: stale verification blocks nothing.
            assert payload["blocked"] is False
            assert client.get("/batches/1/download.csv").status_code == 200
        else:
            # Verification first: it blocked the *old* run and queued the
            # rebuild; once the rebuild publishes, re-verification restores.
            assert verify_outcome["blocked"] is True
            assert verify_outcome["run_id"] == old_run
            # After the racing rebuild finishes, a fresh verification must pass.
            fresh = service.verify_published_batch(db_path, 1)
            assert fresh["status"] == "passed" and fresh["run_id"] == new_published
            assert client.get("/api/batches/1").json()["blocked"] is False

        # Invariant: an active block, if any, always points at the run it found bad.
        with db.connection(db_path) as c:
            block = service.get_active_block(c, 1)
            if block is not None:
                assert block["run_id"] == old_run


def test_verify_all_covers_every_published_batch_and_lists_run_ids(tmp_path):
    db_path = str(tmp_path / "forest.db")
    client, app = make_client(tmp_path)
    with client:
        client.post("/maps", files={"file": ("m.geojson", geojson({"A": (0, 0, 1, 1)}), "application/geo+json")})
        for idx in range(3):
            client.post(
                "/batches",
                files={"file": (f"h{idx}.csv", csv_bytes([(f"H{idx}", 0.5, 0.5, "2026-01-01T00:00:00Z")]), "text/csv")},
            )
        drain(app)
        # Corrupt batch 2 only.
        with db.connection(db_path) as c:
            with db.write_tx(c):
                run2 = c.execute(
                    "SELECT published_run_id FROM batches WHERE id = 2"
                ).fetchone()["published_run_id"]
                c.execute("DELETE FROM results WHERE run_id = ?", (run2,))

        r = client.post("/verify-all", follow_redirects=False)
        assert r.status_code == 303
        statuses = {bid: client.get(f"/api/batches/{bid}").json() for bid in (1, 2, 3)}
        assert statuses[1]["blocked"] is False
        assert statuses[2]["blocked"] is True
        assert statuses[3]["blocked"] is False
        # Healthy batches keep downloading; the bad one is refused.
        assert client.get("/batches/1/download.csv").status_code == 200
        assert client.get("/batches/3/download.csv").status_code == 200
        assert client.get("/batches/2/download.csv").status_code == 409


def test_verification_without_published_result_is_an_error(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        # No batch at all: the route reports the service error via redirect.
        r = client.post("/batches/1/verify", follow_redirects=False)
        assert r.status_code == 303 and "err=" in r.headers["location"]

    db_path = app.state.settings["db_path"]
    with pytest.raises(service.ServiceError):
        service.verify_published_batch(db_path, 999)


def test_block_without_active_map_cannot_queue_but_stays_blocked(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute("DELETE FROM results WHERE run_id = ? AND hotspot_seq = 1", (published_run,))
                # Deactivate every map version so no rebuild can be queued
                # (the rows stay to satisfy the run foreign key).
                c.execute("UPDATE map_versions SET is_active = 0")
        summary = service.verify_published_batch(db_path, 1)
        assert summary["status"] == "failed"
        assert summary["blocked"] is True
        assert summary["requeued_run_id"] is None
        assert client.get("/batches/1/download.csv").status_code == 409


def test_block_reason_and_history_are_persisted(tmp_path):
    client, app = publish_one_batch(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        published_run = client.get("/api/batches/1").json()["published_run_id"]
        with db.connection(db_path) as c:
            with db.write_tx(c):
                c.execute("DELETE FROM results WHERE run_id = ?", (published_run,))
        client.post("/batches/1/verify", follow_redirects=False)
        detail = client.get("/batches/1")
        assert detail.status_code == 200
        assert "供数已封禁" in detail.text
        assert f"run #{published_run}" in detail.text
        assert "核验记录" in detail.text
        with db.connection(db_path) as c:
            verifications = service.list_verifications(c, 1)
        assert verifications[0]["status"] == "failed"
        assert verifications[0]["issues"]
        assert verifications[0]["note"]
