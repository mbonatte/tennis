"""add repeatable render outputs

Revision ID: 20260719_0004
Revises: 20260719_0003
"""
from alembic import op
import sqlalchemy as sa

revision = "20260719_0004"
down_revision = "20260719_0003"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("render_outputs", sa.Column("id", sa.Integer, primary_key=True), sa.Column("public_id", sa.String(36), nullable=False, unique=True), sa.Column("analysis_id", sa.Integer, sa.ForeignKey("analysis_jobs.id"), nullable=False), sa.Column("status", sa.Enum("uploaded","queued","running","completed","failed","cancelled", name="jobstatus", create_type=False), nullable=False), sa.Column("visualization_options", sa.JSON, nullable=False), sa.Column("output_relative_path", sa.String(500)), sa.Column("progress", sa.Integer, nullable=False), sa.Column("current_stage", sa.String(80), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))

def downgrade():
    op.drop_table("render_outputs")
