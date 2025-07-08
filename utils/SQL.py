import sqlite3 as lite
import threading
import numpy as np
lite.register_adapter(np.int64, lambda val: int(val))
lite.register_adapter(np.int32, lambda val: int(val))

def PyDTstoSQLiteDTs(type):
    if type == int: return 'INT',
    if type == str: return 'TEXT',
    if type == float: return 'REAL'

def SQLiteDTstoPyDTs(type):
    if type == 'INT': return int
    if type == 'TEXT': return str
    if type == 'REAL' or type == 'R': return float

def PyValstoSQLiteVals(val, type):
    if val == None: return 'NULL'
    if type == str: return "'" + val + "'"
    return str(val)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

class SQL:
    def __init__(self, dbname, dbdir, tables=None, verbose=False):
        self.dbdir = dbdir
        self.dbname = f"{dbdir}/{dbname}"
        self.verbose = verbose
        self.tables = tables
        if self.tables is None: 
            self.tables = self.get_schema()

        assert len(list(set([x['name'] for x in self.tables]))) == len(self.tables)
        self.write_lock = threading.Lock()
        self.create_tables()

    def __str__(self):
        return f"{self.__class__.__name__}: {self.dbname}"
    
    def get_schema(self):
        tables = []
        table_names = self.query_from_db("SELECT name FROM sqlite_master WHERE type = 'table'")
        for table_name in table_names: 
            name = table_name['name']
            table = {"name": name, "fields": [], "types": [], "primary": []}
            columns = self.query_from_db(f"PRAGMA table_info('{name}')")
            for column in columns: 
                table['fields'].append(column['name'])
                table['types'].append(SQLiteDTstoPyDTs(column['type']))
                if column['pk']: table['primary'].append(column['name'])

            tables.append(table)
    
        return tables

    def execute_query(self, query):
        if self.verbose:
            print("Executing", query, " to db ", self.dbname)
        conn = lite.connect(self.dbname)
        with conn:
            cursor = conn.cursor()
            cursor.execute(query)
        conn.close()

    def param_execution(self, base, data):
        if self.verbose:
            print("Creating parametrized query for", base)
        conn = lite.connect(self.dbname)
        try:
            cursor = conn.cursor()
            for row in data:
                ps = cursor.execute(base, row)
            conn.commit()
        except lite.Error as e:
            if conn:
                conn.rollback()
            raise Exception("Error %s:" % e.args[0])
        finally:
            if conn:
                conn.close()

    def query_from_db(self, query):
        conn = lite.connect(self.dbname)
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def find_table_by_name(self, name):
        table = [x for x in self.tables if x['name'] == name]
        if len(table) != 1:
            raise Exception("Could not find the exact table for name: {}".format(name))
        return table[0]

    def create_table_query(self, table):
        primary_key = ''
        if len(table['primary']):
            primary_key = ", PRIMARY KEY (" + ", ".join(table['primary']) + ")"
        query = 'CREATE TABLE IF NOT EXISTS ' + table['name'] + "(" + \
            ", ".join([x[0] + " " + PyDTstoSQLiteDTs(x[1])[0] for x in zip(table['fields'], table['types'])]) +\
            primary_key + ");"
        return query

    def create_tables(self):
        self.write_lock.acquire()
        for table in self.tables:
            self.execute_query(self.create_table_query(table))
        self.write_lock.release()

    def drop_table(self, name):
        table = self.find_table_by_name(name)
        self.write_lock.acquire()
        self.execute_query('DROP TABLE IF EXISTS {}'.format(table['name']))
        self.write_lock.release()

    def drop_all_tables(self):
        for table in self.tables: self.drop_table(table)

    def insert_into_table(self, name, data):
        table = self.find_table_by_name(name)
        self.write_lock.acquire()
        try:
            self.param_execution(
                "INSERT OR REPLACE INTO " + table['name'] + " VALUES (" + ", ".join([":"+x for x in table['fields']]) + ")",
                data
            )
        except Exception as e:
            raise e
        finally:
            self.write_lock.release()

    def qu_func(self, k, v, tp):
        middle = "="
        end = v
        if type(v) == list:
            end = v[0]
            if v[0] is None and v[1]:
                middle = " IS "
            elif v[0] is None and not v[1]:
                middle = " IS NOT "
            elif not v[1]:
                middle = "!="
        else:
            if v is None: middle = " IS "

        return "{}{}{}".format(k, middle, PyValstoSQLiteVals(end, tp))

    #If you want not null or not a particular value provide an array [val, False]
    def get_data_from_table(self, name, what="*", where=None, group_by=None, order_by= None, order_dir = None, limit = None):
        table = self.find_table_by_name(name)
        query = "SELECT " + what + " FROM " + table['name']
        if where is not None:
            query+= " WHERE "
        else: where = {}
        where_list = []
        for key in where.keys():
            tp = [x[1] for x in zip(table['fields'], table['types']) if x[0] == key][0]
            if type(where[key]) != list:
                where_list.append(self.qu_func(key, where[key], tp))
            elif len(where[key]) == 0: continue
            elif len(where[key]) == 1:
                where_list.append(self.qu_func(key, where[key][0], tp))
            else: where_list.append("(" + " OR ".join([self.qu_func(key, x, tp) for x in where[key]]) + ")")

        query += " AND ".join(where_list)
        if group_by is not None: query += " GROUP BY " + group_by
        if order_by is not None: query += " ORDER BY " + order_by + ((" " + order_dir) if order_dir is not None else "")
        if limit is not None: query += " LIMIT {}".format(limit)
        self.write_lock.acquire()
        data = self.query_from_db(query)
        self.write_lock.release()
        return data
    
    def make_query_dump(self, query, table, filename):
        assert query.startswith("SELECT ")
        conn = lite.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute(f'attach database "{self.dbname}" as adb')
        cursor.execute(f'select sql from adb.sqlite_master where type="table" and name="{table}"')
        sql_create_table = cursor.fetchone()[0]
        cursor.execute(sql_create_table)
        cursor.execute(f'insert into {table} {query.replace(table, f"adb.{table}")}')
        conn.commit()
        cursor.execute("detach database adb")
        with open(f"{self.dbdir}/{filename}", "w") as file:
            file.writelines("\n".join(conn.iterdump()))

    def get_from_dump(self, filename):
        with open(f"{self.dbdir}/{filename}", "r") as file:
            lines = file.readlines()
        conn = lite.connect(self.dbname)
        cursor = conn.cursor()
        for i, l in enumerate(lines):
            l = l.replace("\n", "")
            if l.startswith("CREATE TABLE"): continue
            if l.startswith("INSERT INTO"):
                l = l.replace("INSERT", "INSERT OR REPLACE")
            elif not(l.startswith("BEGIN TRANSACTION;") or l.startswith("COMMIT;")): 
                print(l)
                raise Exception("Danger!")
            cursor.execute(l)
        conn.commit()        

