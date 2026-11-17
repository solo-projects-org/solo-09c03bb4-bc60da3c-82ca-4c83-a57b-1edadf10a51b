"""Cross-process concurrency tests: real Uvicorn servers, one database.

These exercise the guarantee that matters in deployment: two web instances
uploading/recomputing at the same time must never publish a detail file that
mixes results from two different compartment versions.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    def __init__(self, db_path: Path, port: int, cwd: Path):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env.update(
            FOREST_DB_PATH=str(db_path),
            FOREST_RUN_WORKER="1",
            FOREST_STALE_RUN_SECONDS="2",
            PYTHONPATH=str(ROOT),
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 25.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.base + "/healthz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()


def upload_map(client: httpx.Client, name: str, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/maps",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def upload_hotspots(client: httpx.Client, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/batches",
        files={"file": ("hotspots.csv", payload, "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def wait_published(client: httpx.Client, batch_id: int, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"{client.base_url}/api/batches/{batch_id}").json()
        if payload["published_total"] is not None and not payload["pending"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("batch never published")


@pytest.mark.slow
def test_two_servers_publish_single_consistent_version(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            first = wait_published(client_a, 1)
            assert first["published_map_version"] == 1

            # Instance A swaps the map while instance B triggers a recompute.
            upload_map(client_a, "v2.geojson", (SAMPLES / "compartments_v2.geojson").read_bytes())
            response = client_b.post(
                f"{client_b.base_url}/batches/1/recompute", follow_redirects=False
            )
            assert response.status_code == 303

            final = wait_published(client_b, 1)
            assert final["published_map_version"] == 2
            assert final["published_total"] == 15

            # The published CSV must be internally consistent: it is produced
            # exclusively from the run pinned to the active v2 map.
            csv_bytes = client_a.get(f"{client_a.base_url}/batches/1/download.csv").content
            text = csv_bytes.decode("utf-8-sig")
            rows = text.splitlines()
            assert len(rows) == 16
            assert all("v2" in line for line in rows[1:])
            assert "v1" not in text
            assert "A-01;B-02" not in text  # v1-only shared edge between A and B

            # Every hotspot row present exactly once.
            assert text.count("H-00") == 15

            # No computation against the retired v1 map may remain pending, and
            # no v1 run may be the published one.
            payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
            active_version = client_a.get(f"{client_a.base_url}/healthz").json()[
                "active_map_version"
            ]
            assert payload["published_map_version"] == active_version == 2
            for run in payload["runs"]:
                if run["map_version_id"] == 1:
                    assert run["status"] in {"succeeded", "superseded"}
            published_runs = [
                run for run in payload["runs"] if run["id"] == payload["published_run_id"]
            ]
            assert published_runs[0]["map_version_id"] == active_version
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_concurrent_map_swaps_leave_exactly_one_active(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_hotspots(client_a, (SAMPLES / "hotspots.csv").read_bytes())
            v1 = (SAMPLES / "compartments_v1.geojson").read_bytes()
            v2 = (SAMPLES / "compartments_v2.geojson").read_bytes()

            import threading

            errors: list[BaseException] = []

            def swap(client, payload, name):
                try:
                    upload_map(client, name, payload)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=swap, args=(client_a, v1, "a.geojson")),
                threading.Thread(target=swap, args=(client_b, v2, "b.geojson")),
                threading.Thread(target=swap, args=(client_a, v2, "c.geojson")),
                threading.Thread(target=swap, args=(client_b, v1, "d.geojson")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors

            health = client_a.get(f"{client_a.base_url}/healthz").json()
            assert health["active_map_version"] is not None
            from app import db, service

            with db.connection(str(db_path)) as conn:
                active_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM map_versions WHERE is_active = 1"
                ).fetchone()["n"]
                versions = conn.execute("SELECT COUNT(*) AS n FROM map_versions").fetchone()["n"]
            assert active_count == 1
            assert versions == 4

            # The active version is the one the health endpoint reports, and a
            # recompute publishes against it with no mixed output.
            client_b.post(f"{client_b.base_url}/batches/1/recompute", follow_redirects=False)
            final = wait_published(client_b, 1)
            assert final["published_map_version"] == health["active_map_version"]
    finally:
        server_a.stop()
        server_b.stop()


def _corrupt_published_results(db_path: Path, batch_id: int, seqs: list[int]) -> int:
    """Delete result rows directly, as a bad backup restore might."""

    from app import db

    with db.connection(str(db_path)) as conn:
        with db.write_tx(conn):
            run_id = int(
                conn.execute(
                    "SELECT published_run_id FROM batches WHERE id = ?", (batch_id,)
                ).fetchone()["published_run_id"]
            )
            conn.executemany(
                "DELETE FROM results WHERE run_id = ? AND hotspot_seq = ?",
                [(run_id, seq) for seq in seqs],
            )
    return run_id


@pytest.mark.slow
def test_verification_blocks_bad_result_and_recovers_after_rebuild(tmp_path):
    """Two real instances share block state and the rebuild queue end to end."""

    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            wait_published(client_a, 1)

            # A healthy detail verifies cleanly on either instance; downloads
            # keep working normally.
            response = client_a.post(f"{client_a.base_url}/batches/1/verify", follow_redirects=False)
            assert response.status_code == 303
            assert client_a.get(f"{client_a.base_url}/api/batches/1").json()["blocked"] is False
            assert client_b.get(f"{client_b.base_url}/batches/1/download.csv").status_code == 200

            # Corrupt the restored data: two source rows lose their results.
            bad_run = _corrupt_published_results(db_path, 1, [0, 1])
            response = client_b.post(f"{client_b.base_url}/batches/1/verify", follow_redirects=False)
            assert response.status_code == 303

            # The block is visible on BOTH instances and downloads are refused.
            for client in (client_a, client_b):
                payload = client.get(f"{client.base_url}/api/batches/1").json()
                assert payload["blocked"] is True
                assert payload["verifications"][0]["status"] == "failed"
                assert payload["verifications"][0]["run_id"] == bad_run
                blocked = client.get(f"{client.base_url}/batches/1/download.csv")
                assert blocked.status_code == 409

            # The verification enqueued a rebuild; one of the two workers picks
            # it up and publishes, but the block survives until re-verified.
            deadline = time.time() + 30
            while time.time() < deadline:
                payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
                if payload["published_run_id"] != bad_run and not payload["pending"]:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("rebuild never published")
            assert payload["blocked"] is True
            assert client_a.get(f"{client_a.base_url}/batches/1/download.csv").status_code == 409

            # Re-verification of the new result passes on instance A and
            # restores downloads on instance B as well.
            response = client_a.post(f"{client_a.base_url}/batches/1/verify", follow_redirects=False)
            assert response.status_code == 303
            payload_a = client_a.get(f"{client_a.base_url}/api/batches/1").json()
            payload_b = client_b.get(f"{client_b.base_url}/api/batches/1").json()
            assert payload_a["blocked"] is False and payload_b["blocked"] is False
            csv_text = client_b.get(f"{client_b.base_url}/batches/1/download.csv").content.decode(
                "utf-8-sig"
            )
            assert csv_text.count("H-00") == 15
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_verify_all_on_two_instances_shares_block_state(tmp_path):
    """verify-all triggered on one instance blocks downloads seen on the other."""

    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            wait_published(client_a, 1)
            bad_run = _corrupt_published_results(db_path, 1, [3])

            response = client_a.post(f"{client_a.base_url}/verify-all", follow_redirects=False)
            assert response.status_code == 303

            payload = client_b.get(f"{client_b.base_url}/api/batches/1").json()
            assert payload["blocked"] is True
            assert payload["verifications"][0]["run_id"] == bad_run
            assert client_b.get(f"{client_b.base_url}/batches/1/download.csv").status_code == 409

            # Rebuild and recover; then a healthy verify-all reports nothing bad.
            deadline = time.time() + 30
            while time.time() < deadline:
                payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
                if payload["published_run_id"] != bad_run and not payload["pending"]:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("rebuild never published")
            client_b.post(f"{client_b.base_url}/verify-all", follow_redirects=False)
            assert client_a.get(f"{client_a.base_url}/api/batches/1").json()["blocked"] is False
            assert client_a.get(f"{client_a.base_url}/batches/1/download.csv").status_code == 200
    finally:
        server_a.stop()
        server_b.stop()
