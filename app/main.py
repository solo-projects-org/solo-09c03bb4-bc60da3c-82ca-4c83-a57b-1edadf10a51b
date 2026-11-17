"""FastAPI application: uploads, run status, conflict review, CSV download."""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__, db, service
from .parsing import InputError
from .worker import Worker

BASE_DIR = Path(__file__).resolve().parent
log = logging.getLogger("forest.web")


def _settings() -> dict[str, Any]:
    return {
        "db_path": os.environ.get("FOREST_DB_PATH", str(BASE_DIR.parent / "data" / "forest.db")),
        "upload_max_mb": int(os.environ.get("FOREST_UPLOAD_MAX_MB", "32")),
        "run_worker": os.environ.get("FOREST_RUN_WORKER", "1") not in {"0", "false", "no"},
        "stale_seconds": int(os.environ.get("FOREST_STALE_RUN_SECONDS", "120")),
        "per_page": int(os.environ.get("FOREST_PREVIEW_ROWS", "200")),
    }


def create_app(db_path: str | None = None, start_worker: bool | None = None) -> FastAPI:
    settings = _settings()
    settings["db_path"] = db_path or settings["db_path"]
    if start_worker is not None:
        settings["run_worker"] = start_worker

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(app.state.settings["db_path"])
        worker = None
        if app.state.settings["run_worker"]:
            worker = Worker(
                app.state.settings["db_path"],
                stale_seconds=app.state.settings["stale_seconds"],
            )
            worker.start()
        app.state.worker = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()

    app = FastAPI(title="林火卫星热点林班归属", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.worker = None
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["status_label"] = lambda value: service.STATUS_LABELS.get(value, value)
    templates.env.filters["reason_label"] = lambda value: service.REASON_LABELS.get(value, value)
    templates.env.filters["run_status_label"] = lambda value: service.RUN_STATUS_LABELS.get(value, value)

    def conn() -> sqlite3.Connection:
        return db.connect(app.state.settings["db_path"])

    def kick_worker() -> None:
        worker = app.state.worker
        if worker is not None:
            worker.kick()

    def render(request: Request, template: str, context: dict[str, Any], status_code: int = 200):
        return templates.TemplateResponse(
            request=request, name=template, context=context, status_code=status_code
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, msg: str | None = None, err: str | None = None):
        with conn() as c:
            active_map = service.get_active_map(c)
            batches = service.list_batches(c)
            maps = service.list_maps(c, limit=10)
        return render(
            request,
            "index.html",
            {
                "active_map": active_map,
                "maps": maps,
                "batches": batches,
                "msg": msg,
                "err": err,
                "max_mb": app.state.settings["upload_max_mb"],
            },
        )

    @app.post("/maps")
    def upload_map(request: Request, file: UploadFile = File(...)):
        try:
            data = _read_upload(file, app.state.settings["upload_max_mb"], "林班 GeoJSON")
            info, stale_batches = service.publish_map(
                app.state.settings["db_path"], data, _safe_filename(file.filename, "compartments.geojson")
            )
        except InputError as exc:
            return _redirect("/", err=str(exc))
        note = f"林班底图已发布为 v{info.id}（{info.feature_count} 个林班）"
        if info.repaired_count:
            note += f"，其中 {info.repaired_count} 个无效几何已自动修复"
        if stale_batches:
            note += f"。有 {len(stale_batches)} 个批次的已发布结果基于旧底图，请重算"
        return _redirect("/", msg=note)

    @app.post("/recompute-all")
    def recompute_all(request: Request):
        with conn() as c:
            active = service.get_active_map(c)
            if active is None:
                return _redirect("/", err="尚未上传林班底图")
            batches = [row["id"] for row in c.execute("SELECT id FROM batches ORDER BY id").fetchall()]
        queued = 0
        errors = []
        for batch_id in batches:
            try:
                service.enqueue_recompute(app.state.settings["db_path"], int(batch_id))
                queued += 1
            except service.ServiceError as exc:
                errors.append(str(exc))
        kick_worker()
        note = f"已为 {queued} 个批次按 v{active.id} 重新排队计算"
        if errors:
            return _redirect("/", msg=note, err="；".join(errors))
        return _redirect("/", msg=note)

    @app.post("/verify-all")
    def verify_all(request: Request):
        """Post-restore check of every published detail, across both instances."""

        with conn() as c:
            targets = [
                int(row["id"])
                for row in c.execute(
                    "SELECT id FROM batches WHERE published_run_id IS NOT NULL ORDER BY id"
                ).fetchall()
            ]
        if not targets:
            return _redirect("/", err="当前没有已发布的结果可供核验")
        summaries = []
        for batch_id in targets:
            try:
                summaries.append(service.verify_published_batch(app.state.settings["db_path"], batch_id))
            except service.ServiceError as exc:
                return _redirect("/", err=str(exc))
        kick_worker()
        failed = [item for item in summaries if item.get("applied") and item["status"] == service.VERIFY_FAILED]
        blocked = sum(1 for item in failed if item["blocked"])
        recovered = sum(1 for item in summaries if item.get("recovered"))
        note = f"已核验 {len(summaries)} 个批次，受检计算号：" + "、".join(
            f"#{item['run_id']}" for item in summaries
        )
        err = None
        if failed:
            err = f"{blocked} 个批次发现缺行/额外行/统计不符，已封禁下载并按当前底图排队重建"
        if recovered:
            note += f"；{recovered} 个批次重新核验合格，已恢复供数"
        return _redirect("/", msg=note, err=err)

    @app.post("/batches/{batch_id}/verify")
    def verify_batch(request: Request, batch_id: int):
        try:
            summary = service.verify_published_batch(app.state.settings["db_path"], batch_id)
        except service.ServiceError as exc:
            return _redirect(f"/batches/{batch_id}", err=str(exc))
        kick_worker()
        if not summary["applied"]:
            return _redirect(
                f"/batches/{batch_id}",
                err="核验期间该批次已被其他实例重新发布，本次核验结果未应用（不影响新结果）",
            )
        run_no = summary["run_id"]
        if summary["status"] == service.VERIFY_PASSED:
            if summary["recovered"]:
                msg = f"受检计算号 #{run_no} 核验合格，已恢复下载供数"
            else:
                msg = f"受检计算号 #{run_no} 核验合格：每来源行恰好一条，状态汇总与发布统计一致"
            return _redirect(f"/batches/{batch_id}", msg=msg)
        kinds = "、".join(issue["label"] for issue in summary["issues"][:4])
        err = (
            f"受检计算号 #{run_no} 核验不合格：{kinds}；已封禁该结果后续下载"
        )
        if summary["requeued_run_id"] is not None:
            err += f"，并按当前底图排队重建（run #{summary['requeued_run_id']}）"
        else:
            err += "，但当前无活动底图，未能排队重建"
        return _redirect(f"/batches/{batch_id}", err=err)

    @app.post("/batches")
    def upload_batch(request: Request, file: UploadFile = File(...)):
        try:
            data = _read_upload(file, app.state.settings["upload_max_mb"], "热点 CSV")
            batch_id, run_id = service.create_batch(
                app.state.settings["db_path"], data, _safe_filename(file.filename, "hotspots.csv")
            )
        except InputError as exc:
            return _redirect("/", err=str(exc))
        if run_id is None:
            return _redirect("/", msg=f"热点批次 #{batch_id} 已保存；上传林班底图后即可计算")
        kick_worker()
        return _redirect(f"/batches/{batch_id}", msg=f"热点批次 #{batch_id} 已上传，计算已排队")

    @app.post("/batches/{batch_id}/recompute")
    def recompute(request: Request, batch_id: int):
        try:
            run_id = service.enqueue_recompute(app.state.settings["db_path"], batch_id)
        except service.ServiceError as exc:
            return _redirect(f"/batches/{batch_id}", err=str(exc))
        kick_worker()
        return _redirect(
            f"/batches/{batch_id}", msg=f"已按当前底图排队重算（计算记录 #{run_id}）"
        )

    @app.get("/batches/{batch_id}", response_class=HTMLResponse)
    def batch_detail(
        request: Request,
        batch_id: int,
        msg: str | None = None,
        err: str | None = None,
        only: str | None = None,
    ):
        with conn() as c:
            batch = service.get_batch(c, batch_id)
            if batch is None:
                raise HTTPException(status_code=404, detail="热点批次不存在")
            active_map = service.get_active_map(c)
            published = service.get_published_run(c, batch)
            runs = service.list_runs(c, batch_id)
            block = service.get_active_block(c, batch_id)
            verifications = service.list_verifications(c, batch_id)
            counts = service.result_counts(c, published["id"]) if published else []
            preview = (
                service.preview_results(
                    c,
                    published["id"],
                    limit=app.state.settings["per_page"],
                    only_problems=only == "problems",
                )
                if published
                else []
            )
            pending = service.is_run_active(c, batch_id)
        return render(
            request,
            "batch.html",
            {
                "batch": batch,
                "active_map": active_map,
                "published": published,
                "runs": runs,
                "block": block,
                "verifications": verifications,
                "counts": counts,
                "preview": preview,
                "only_problems": only == "problems",
                "pending": pending,
                "msg": msg,
                "err": err,
                "per_page": app.state.settings["per_page"],
            },
        )

    @app.get("/batches/{batch_id}/download.csv")
    def download(batch_id: int):
        with conn() as c:
            batch = service.get_batch(c, batch_id)
            if batch is None:
                raise HTTPException(status_code=404, detail="热点批次不存在")
            block = service.get_active_block(c, batch_id)
            published = service.get_published_run(c, batch)
        if block is not None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"批次 #{batch_id} 的已发布结果 run #{block['run_id']} 核验不合格，"
                    "下载已封禁；按当前底图重建并重新核验合格后自动恢复"
                ),
            )
        if published is None:
            raise HTTPException(status_code=409, detail="该批次还没有可下载的计算结果")
        name = f"hotspot_attribution_batch{batch_id}_run{published['id']}.csv"
        headers = {
            "Content-Disposition": f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}",
            "Cache-Control": "no-store",
        }
        return StreamingResponse(
            service.iter_result_csv(app.state.settings["db_path"], batch_id),
            media_type="text/csv; charset=utf-8",
            headers=headers,
        )

    @app.get("/api/batches/{batch_id}")
    def batch_status(batch_id: int):
        """Small JSON view, useful for polling and for smoke tests."""

        with conn() as c:
            batch = service.get_batch(c, batch_id)
            if batch is None:
                raise HTTPException(status_code=404, detail="热点批次不存在")
            published = service.get_published_run(c, batch)
            runs = service.list_runs(c, batch_id, limit=10)
            block = service.get_active_block(c, batch_id)
            verifications = service.list_verifications(c, batch_id, limit=5)
        return {
            "batch_id": batch_id,
            "published_run_id": batch["published_run_id"],
            "published_map_version": published["map_version_id"] if published else None,
            "published_total": published["total_count"] if published else None,
            "matched": published["matched_count"] if published else None,
            "conflict": published["conflict_count"] if published else None,
            "unmatched": published["unmatched_count"] if published else None,
            "blocked": block is not None,
            "block": block,
            "verifications": [
                {
                    "id": item["id"],
                    "run_id": item["run_id"],
                    "map_version_id": item["map_version_id"],
                    "status": item["status"],
                    "applied": bool(item["applied"]),
                    "issue_count": item["issue_count"],
                    "issues": item["issues"],
                    "note": item["note"],
                    "checked_at": item["checked_at"],
                }
                for item in verifications
            ],
            "pending": any(
                run["status"] in (service.STATUS_QUEUED, service.STATUS_RUNNING) for run in runs
            ),
            "runs": [
                {
                    "id": run["id"],
                    "status": run["status"],
                    "map_version_id": run["map_version_id"],
                    "matched": run["matched_count"],
                    "conflict": run["conflict_count"],
                    "unmatched": run["unmatched_count"],
                    "error": run["error"],
                }
                for run in runs
            ],
        }

    @app.get("/healthz")
    def healthz():
        with conn() as c:
            active = service.get_active_map(c)
        return {"status": "ok", "version": __version__, "active_map_version": active.id if active else None}

    return app


def _read_upload(file: UploadFile, max_mb: int, what: str) -> bytes:
    data = file.file.read()
    if not data:
        raise InputError(f"{what} 为空")
    if len(data) > max_mb * 1024 * 1024:
        raise InputError(f"{what} 超过 {max_mb} MB 限制")
    return data


def _safe_filename(name: str | None, fallback: str) -> str:
    text = Path(name or "").name.strip()
    return text or fallback


def _redirect(path: str, msg: str | None = None, err: str | None = None) -> RedirectResponse:
    from urllib.parse import urlencode

    params = {key: value for key, value in (("msg", msg), ("err", err)) if value}
    url = path + (("?" + urlencode(params)) if params else "")
    return RedirectResponse(url, status_code=303)


app = create_app()
