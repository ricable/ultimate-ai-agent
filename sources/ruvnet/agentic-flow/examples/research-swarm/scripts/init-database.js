#!/usr/bin/env node
import { initDatabase } from '../lib/db-utils.js';

console.log('🔧 Initializing SQLite database...\n');
initDatabase();
console.log('\n✅ Database ready!');
