#!/bin/bash
# Backup script for AI Interview Assistant
# Run this script regularly using cron

set -e

# Configuration
BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="ai_interview_backup_${TIMESTAMP}"
RETENTION_DAYS=30

echo "Starting backup: ${BACKUP_NAME}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup database
echo "Backing up database..."
if [ -n "$DATABASE_URL" ]; then
    if [[ $DATABASE_URL == postgresql* ]]; then
        # PostgreSQL backup
        pg_dump $DATABASE_URL > "${BACKUP_DIR}/${BACKUP_NAME}/database.sql"
    elif [[ $DATABASE_URL == sqlite* ]]; then
        # SQLite backup
        DB_PATH=$(echo $DATABASE_URL | sed 's/sqlite:\/\/\///')
        cp "$DB_PATH" "${BACKUP_DIR}/${BACKUP_NAME}/database.db"
    fi
fi

# Backup configuration (excluding secrets)
echo "Backing up configuration..."
cp .env.example "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup domain configurations
echo "Backing up domain configs..."
cp -r Domains "${BACKUP_DIR}/${BACKUP_NAME}/"

# Backup recent session data (last 7 days)
echo "Backing up recent session data..."
find outputs/ -type f -mtime -7 -exec cp --parents {} "${BACKUP_DIR}/${BACKUP_NAME}/" \;

# Create tarball
echo "Creating compressed archive..."
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

# Calculate size
SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
echo "Backup completed: ${BACKUP_NAME}.tar.gz (${SIZE})"

# Clean old backups
echo "Cleaning backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "ai_interview_backup_*.tar.gz" -mtime +${RETENTION_DAYS} -delete

echo "Backup process completed successfully"

# Optional: Upload to cloud storage
# aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" s3://your-bucket/backups/
# gsutil cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" gs://your-bucket/backups/
