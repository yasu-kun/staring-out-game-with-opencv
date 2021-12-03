# Pythonでデータベースに接続
import sqlite3
# SQLiteのファイルにアクセス
con = sqlite3.connect('database.db')

cur = con.cursor()

#複数行に渡ってSQL文を書く場合は'''で囲む．
cur.execute('''CREATE table cam1(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    angry TEXT,
                    disgust TEXT,
                    fear TEXT,
                    happy TEXT,
                    sad TEXT,
                    surprise TEXT,
                    neutral TEXT
                    );''')

cur.execute('''CREATE table cam2(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    angry TEXT,
                    disgust TEXT,
                    fear TEXT,
                    happy TEXT,
                    sad TEXT,
                    surprise TEXT,
                    neutral TEXT
                    );''')

con.commit()
con.close()


