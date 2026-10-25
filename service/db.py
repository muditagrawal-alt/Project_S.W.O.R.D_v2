"""
Phase 6 storage: runs, incidents, events and an audit log (SQLAlchemy 2.x).

SQLite by default; PostgreSQL works by passing a URL such as postgresql+psycopg://user:pass@host/sword
(install a driver, e.g. `pip install psycopg[binary]`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    source: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="queued")   # queued | running | finished | stopped | failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    out_dir: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="open")     # open | closed (set by the system)
    level_start: Mapped[str] = mapped_column(String(32))
    level_peak: Mapped[str] = mapped_column(String(32))
    start_t: Mapped[float] = mapped_column(Float)
    end_t: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    track_ids: Mapped[list] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    clip_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snapshot_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    clip_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)      # chain of custody
    snapshot_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ack_by: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)           # operator review
    ack_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ack_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    incident_id: Mapped[Optional[int]] = mapped_column(ForeignKey("incidents.id"), nullable=True, index=True)
    t: Mapped[float] = mapped_column(Float)
    frame_idx: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(10))        # track | scene
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    from_level: Mapped[str] = mapped_column(String(32))
    to_level: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(80))
    target: Mapped[str] = mapped_column(String(120))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


def make_session_factory(url: str):
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def to_dict(obj) -> dict:
    out = {}
    for col in obj.__table__.columns:
        v = getattr(obj, col.name)
        out[col.name] = v.isoformat() if isinstance(v, datetime) else v
    return out


def audit(session, actor: str, action: str, target: str, **details) -> None:
    session.add(AuditLog(actor=actor, action=action, target=target, details=details))
