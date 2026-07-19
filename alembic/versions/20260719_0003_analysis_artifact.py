"""store persisted analysis artifact path

Revision ID: 20260719_0003
Revises: 20260719_0002
"""
from alembic import op
import sqlalchemy as sa

revision = "20260719_0003"
down_revision = "20260719_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("analysis_artifact_relative_path", sa.String(length=500)))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "analysis_artifact_relative_path")
