"""Inital db migration

Revision ID: deb339ea450b
Revises:
Create Date: 2022-10-03 22:58:40.818798

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "deb339ea450b"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "batch",
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_id", sa.String(length=50), nullable=True),
        sa.Column("urls", sa.ARRAY(sa.String(length=1000)), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("total_urls", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_batch_batch_id"), "batch", ["batch_id"], unique=True)
    op.create_index(op.f("ix_batch_id"), "batch", ["id"], unique=False)
    op.create_table(
        "batch_result",
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_id", sa.String(length=50), nullable=False),
        sa.Column("index", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(length=1000), nullable=False),
        sa.Column("result", postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["batch_id"],
            ["batch.batch_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_batch_result_batch_id"), "batch_result", ["batch_id"], unique=False
    )
    op.create_index(op.f("ix_batch_result_id"), "batch_result", ["id"], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_batch_result_id"), table_name="batch_result")
    op.drop_index(op.f("ix_batch_result_batch_id"), table_name="batch_result")
    op.drop_table("batch_result")
    op.drop_index(op.f("ix_batch_id"), table_name="batch")
    op.drop_index(op.f("ix_batch_batch_id"), table_name="batch")
    op.drop_table("batch")
    # ### end Alembic commands ###
