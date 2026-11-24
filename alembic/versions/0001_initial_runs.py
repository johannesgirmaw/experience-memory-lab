"""initial runs / episodes / steps / memory_records

Revision ID: 0001
Revises:
Create Date: 2026-11-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("condition", sa.String(length=8), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("env_version", sa.String(length=64), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=False),
        sa.Column("git_hash", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("episode_index", sa.Integer(), nullable=False),
        sa.Column("phase", sa.String(length=8), nullable=False),
        sa.Column("hidden_rule_id", sa.String(length=64), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("total_reward", sa.Float(), nullable=True),
        sa.Column("n_steps", sa.Integer(), nullable=True),
    )
    op.create_index("ix_episodes_run_id", "episodes", ["run_id"])
    op.create_table(
        "steps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("episode_id", sa.String(length=64), sa.ForeignKey("episodes.id"), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("action_json", sa.JSON(), nullable=False),
        sa.Column("reward", sa.Float(), nullable=False),
        sa.Column("observation_json", sa.JSON(), nullable=False),
        sa.Column("info_json", sa.JSON(), nullable=False),
        sa.Column("terminated", sa.Boolean(), nullable=False),
        sa.Column("truncated", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_steps_episode_id", "steps", ["episode_id"])
    op.create_table(
        "memory_records",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("condition", sa.String(length=8), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("body_json", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_memory_records_run_id", "memory_records", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_memory_records_run_id", table_name="memory_records")
    op.drop_table("memory_records")
    op.drop_index("ix_steps_episode_id", table_name="steps")
    op.drop_table("steps")
    op.drop_index("ix_episodes_run_id", table_name="episodes")
    op.drop_table("episodes")
    op.drop_table("runs")
