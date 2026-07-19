"""store user court calibration

Revision ID: 20260719_0005
Revises: 20260719_0004
"""

import sqlalchemy as sa

from alembic import op

revision = "20260719_0005"
down_revision = "20260719_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("court_calibration", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "court_calibration")
