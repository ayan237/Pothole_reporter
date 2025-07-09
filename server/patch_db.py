import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Add the image_filename column if it doesn't exist
try:
    c.execute("ALTER TABLE reports ADD COLUMN image_filename TEXT")
    print("✅ Column 'image_filename' added successfully.")
except sqlite3.OperationalError:
    print("⚠️ Column 'image_filename' already exists.")

conn.commit()
conn.close()
