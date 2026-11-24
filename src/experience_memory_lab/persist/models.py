"""SQLAlchemy models for runs. No User/Org/ApiKey tables — this is a lab."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


# JSONB on Postgres, plain JSON elsewhere (sqlite tests later).
JsonType = JSON().with_variant(JSONB(), "postgresql")


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    condition: Mapped[str] = mapped_column(String(8), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    env_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[dict] = mapped_column(JsonType, nullable=False, default=dict)
    git_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    episodes: Mapped[list[EpisodeRow]] = relationship(back_populates="run")


class EpisodeRow(Base):
    __tablename__ = "episodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    episode_index: Mapped[int] = mapped_column(Integer, nullable=False)
    phase: Mapped[str] = mapped_column(String(8), nullable=False)
    hidden_rule_id: Mapped[str] = mapped_column(String(64), nullable=False)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    total_reward: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)

    run: Mapped[RunRow] = relationship(back_populates="episodes")
    steps: Mapped[list[StepRow]] = relationship(back_populates="episode")


class StepRow(Base):
    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.id"), nullable=False, index=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    action_json: Mapped[dict] = mapped_column(JsonType, nullable=False, default=dict)
    reward: Mapped[float] = mapped_column(Float, nullable=False)
    observation_json: Mapped[dict] = mapped_column(JsonType, nullable=False, default=dict)
    info_json: Mapped[dict] = mapped_column(JsonType, nullable=False, default=dict)
    terminated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    episode: Mapped[EpisodeRow] = relationship(back_populates="steps")


# Placeholder for later memory tables (B/C/D). Keep migrations additive.
class MemoryRecordRow(Base):
    __tablename__ = "memory_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    condition: Mapped[str] = mapped_column(String(8), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # note|skill|experience
    body_json: Mapped[dict] = mapped_column(JsonType, nullable=False, default=dict)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
