# Pythonでデータベースに接続
import sqlite3
# SQLiteのファイルにアクセス
con = sqlite3.connect('database.db')

cur = con.cursor()

#複数行に渡ってSQL文を書く場合は'''で囲む．
cur.execute('''SELECT * FROM cam1;''')

li = cur.fetchall()
#print(li)

cur.execute('''SELECT * FROM cam2;''')

li = cur.fetchall()
#print(li)

cur.execute('select max(id), angry,disgust,fear,happy,sad,surprise,neutral from cam1;')

li = cur.fetchall()
print(li)

con.commit()
con.close()


