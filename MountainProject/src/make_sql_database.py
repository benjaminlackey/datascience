"""This script takes the data (csv format) extracted from mountainproject.com,
and puts it into a SQL database. Some additional cleaning is applied.
"""

import pandas as pd
import sqlite3

df_users = pd.read_csv('../data/user_data.csv')
df_items = pd.read_csv('../data/route_data.csv')
df_ratings = pd.read_csv('../data/route_ratings_merged.csv')

conn = sqlite3.connect('../data/mountain_project_gunks.sqlite')
cur = conn.cursor()


# Setup SQL tables to hold information about the
# types of protection, climbers, routes, ratings of each climber.
cur.executescript("""
DROP TABLE IF EXISTS pro;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS ratings;


CREATE TABLE pro (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    pro_type VARCHAR(5)
);

INSERT INTO pro (pro_type) VALUES ('Trad');
INSERT INTO pro (pro_type) VALUES ('TR');
INSERT INTO pro (pro_type) VALUES ('Sport');


CREATE TABLE users (
    uid INTEGER NOT NULL PRIMARY KEY UNIQUE,
    name TEXT,
    user_url TEXT UNIQUE,
    profile TEXT
);


CREATE TABLE items (
    iid INTEGER NOT NULL PRIMARY KEY UNIQUE,
    route_name TEXT,
    route_url TEXT UNIQUE,
    difficulty TEXT,
    pro_id INTEGER,
    length INTEGER,
    pitches INTEGER,
    nratings INTEGER,
    description TEXT,
    monthly_views INTEGER,
    total_views INTEGER
);


CREATE TABLE ratings (
    uid INTEGER,
    iid INTEGER,
    rating INTEGER,
    PRIMARY KEY (uid, iid)
);
""")


# Replace protection name with the unique id to save memory.
def lookup_pro_id(cur, pro_type):
    """Get the integer pro_id from the string pro.
    """
    cur.execute('SELECT id FROM pro WHERE pro_type=?', (pro_type, ))
    return cur.fetchone()[0]


# Fill in the route data. Clean it when necessary.
for i in range(len(df_items)):
    row = df_items.loc[i].to_dict()
    iid = row['iid']
    route_name = row['route_name']
    route_url = row['route_url']
    difficulty = row['difficulty']
    # Replace the protection string with the integer id
    pro_id = lookup_pro_id(cur, row['pro_type'])
    length = row['length']
    pitches = row['pitches']
    nratings = row['nratings']
    # Strip all non ascii characters
    description = row['description'].decode('ascii', errors='ignore').encode()
    monthly_views = row['monthly_views']
    total_views = row['total_views']

    cur.execute("""
    INSERT INTO items (
    iid, route_name, route_url, difficulty, pro_id, length,
    pitches, nratings, description, monthly_views, total_views)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (iid, route_name, route_url, difficulty, pro_id, length,
          pitches, nratings, description, monthly_views, total_views))


# Fill in the climbers data. Clean it when necessary.
for i in range(len(df_users)):
    row = df_users.loc[i].to_dict()
    uid = row['uid']
    name = row['name'].decode('ascii', errors='ignore').encode()
    user_url = row['user_url']
    profile = row['profile']
    # There are some blank profile entries.
    # Strip all non ascii characters.
    try:
        profile = profile.decode('ascii', errors='ignore').encode()
    except:
        profile = None

    cur.execute("""
    INSERT INTO users (uid, name, user_url, profile)
    VALUES (?, ?, ?, ?)
    """, (uid, name, user_url, profile))


# Fill in the ratings table.
for i in range(len(df_ratings)):
    row = df_ratings.loc[i].to_dict()

    cur.execute(
        'INSERT INTO ratings (uid, iid, rating) VALUES (?, ?, ?)',
        (row['uid'], row['iid'], row['rating']))


# Save and close
conn.commit()
conn.close()
