import sqlite3
from html import escape
import subprocess
import shlex

def process_user_input(user_input):
    """Process user input with multiple security vulnerabilities"""
    
    # Security Issue 1: SQL Injection
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_input}"
    cursor.execute(query)
    
    # Use subprocess with shell=False for safe command execution
    subprocess.run(['echo', user_input], shell=False, check=True)
    
    # Security Issue 3: XSS
    html = f"<div>{escape(user_input)}</div>"
    
    return query, html

def validate_input(data):
    """Insecure input validation"""
    # Security Issue 4: Weak validation
    if len(data) > 0:
        return True
    return False

def store_password(password):
    """Insecure password storage"""
    # Security Issue 5: Plain text password
    with open('passwords.txt', 'a') as f:
        f.write(password + '\n')
# Sample vulnerable app code
import sqlite3
import os

def get_user(user_id):
    # SQL Injection vulnerability
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()

def echo_input(user_input):
    # Safe command execution using subprocess
    subprocess.run(['echo', user_input], shell=False, check=True)

def display_comment(comment):
    # Safe HTML output with proper escaping
    return f"<div>{escape(comment)}</div>"
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/users')
def get_users():
    # Vulnerable to SQL injection
    name = request.args.get('name')
    query = f"SELECT * FROM users WHERE name = '{name}'"
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    return str(cursor.fetchall())

if __name__ == '__main__':
    app.run(debug=True)
