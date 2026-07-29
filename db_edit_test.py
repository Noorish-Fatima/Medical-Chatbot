from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic
revision = '12345abcde'
down_revision = None

def upgrade():
    # 1. Execute raw SQL to modify schema
    op.execute("""
        ALTER TABLE users 
        ADD COLUMN status VARCHAR(20) DEFAULT 'active';
    """)

def downgrade():
    # 2. Revert the changes if needed
    op.execute("""
        ALTER TABLE users 
        DROP COLUMN status;
    """)
