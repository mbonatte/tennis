"""add repeatable render outputs

Revision ID: 20260719_0004
Revises: 20260719_0003
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260719_0004"
down_revision = "20260719_0003"
branch_labels = None
depends_on = None


def _existing_job_status_enum() -> postgresql.ENUM:
    """Reference the enum created by the initial jobs migration."""
    return postgresql.ENUM(
        "uploaded",
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
        name="jobstatus",
        create_type=False,
    )


def upgrade() -> None:
    op.create_table(
        "render_outputs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("public_id", sa.String(36), nullable=False, unique=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analysis_jobs.id"), nullable=False),
        sa.Column(
            "status",
            _existing_job_status_enum(),
            nullable=False,
        ),
        sa.Column("visualization_options", sa.JSON(), nullable=False),
        sa.Column("output_relative_path", sa.String(500)),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("current_stage", sa.String(80), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_render_outputs_public_id", "render_outputs", ["public_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_render_outputs_public_id", table_name="render_outputs")
    op.drop_table("render_outputs")
