#!/bin/sh

rm database.db
echo '[INFO] removed db file'

touch database.db
echo '[INFO] generated new db file'

python set_database.py
echo '[INFO] created tables'
