"""persist analysis job workflow progress

Revision ID: 20260719_0002
Revises: 20260712_0001
"""

from alembic import op
import sqlalchemy as sa

revision = "20260719_0002"
down_revision = "20260712_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("workflow", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.alter_column("analysis_jobs", "workflow", server_default=None)


def downgrade() -> None:
    op.drop_column("analysis_jobs", "workflow")
