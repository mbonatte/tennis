"""create analysis jobs

Revision ID: 20260712_0001
"""
from alembic import op
import sqlalchemy as sa

revision = "20260712_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    status = sa.Enum("uploaded", "queued", "running", "completed", "failed", "cancelled", name="jobstatus")
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("public_id", sa.String(36), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False), sa.Column("stored_filename", sa.String(255), nullable=False),
        sa.Column("status", status, nullable=False), sa.Column("current_stage", sa.String(80), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False), sa.Column("submitted_options", sa.JSON(), nullable=False),
        sa.Column("input_relative_path", sa.String(500), nullable=False), sa.Column("output_video_relative_path", sa.String(500)),
        sa.Column("result_relative_path", sa.String(500)), sa.Column("error_type", sa.String(120)),
        sa.Column("error_message", sa.Text()), sa.Column("internal_diagnostic", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)), sa.Column("input_size", sa.BigInteger(), nullable=False),
        sa.Column("output_size", sa.BigInteger()), sa.Column("video_duration", sa.Float()),
        sa.Column("video_width", sa.Integer()), sa.Column("video_height", sa.Integer()), sa.Column("video_codec", sa.String(40)),
        sa.Column("pipeline_version", sa.String(40), nullable=False), sa.Column("queue_job_id", sa.String(80)),
        sa.Column("cancellation_requested", sa.Boolean(), nullable=False), sa.UniqueConstraint("public_id"),
    )
    op.create_index("ix_analysis_jobs_public_id", "analysis_jobs", ["public_id"], unique=True)
    op.create_index("ix_analysis_jobs_status", "analysis_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_status", table_name="analysis_jobs")
    op.drop_index("ix_analysis_jobs_public_id", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
    sa.Enum(name="jobstatus").drop(op.get_bind(), checkfirst=True)
