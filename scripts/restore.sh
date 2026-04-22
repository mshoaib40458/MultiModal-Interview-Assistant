#!/bin/bash
# Restore script for AI Interview Assistant

set -e

# Check if backup file provided
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo "Available backups:"
    ls -lh /backups/*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="/tmp/restore_$$"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "Restoring from: $BACKUP_FILE"
echo "WARNING: This will overwrite existing data!"
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Extract backup
echo "Extracting backup..."
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

# Find extracted directory
EXTRACTED_DIR=$(find "$RESTORE_DIR" -maxdepth 1 -type d -name "ai_interview_backup_*")

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: Invalid backup file structure"
    rm -rf "$RESTORE_DIR"
    exit 1
fi

# Stop services
echo "Stopping services..."
echo "Please stop the application manually before restoring."

# Restore database
echo "Restoring database..."
if [ -f "${EXTRACTED_DIR}/database.sql" ]; then
    # PostgreSQL restore
    psql $DATABASE_URL < "${EXTRACTED_DIR}/database.sql"
elif [ -f "${EXTRACTED_DIR}/database.db" ]; then
    # SQLite restore
    DB_PATH=$(echo $DATABASE_URL | sed 's/sqlite:\/\/\///')
    cp "${EXTRACTED_DIR}/database.db" "$DB_PATH"
fi

# Restore domain configurations
echo "Restoring domain configs..."
if [ -d "${EXTRACTED_DIR}/Domains" ]; then
    rm -rf Domains
    cp -r "${EXTRACTED_DIR}/Domains" .
fi

# Restore session data
echo "Restoring session data..."
if [ -d "${EXTRACTED_DIR}/outputs" ]; then
    cp -r "${EXTRACTED_DIR}/outputs/"* outputs/ 2>/dev/null || true
fi

# Cleanup
rm -rf "$RESTORE_DIR"

# Start services
echo "Starting services..."
echo "Please restart the application manually after restoring."

echo "Restore completed successfully!"
echo "Please verify the system is working correctly."
