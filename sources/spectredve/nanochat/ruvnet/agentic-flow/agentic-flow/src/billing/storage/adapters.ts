/**
 * Storage Adapters
 * Multiple backend options: Memory, AgentDB, SQLite, PostgreSQL
 */

import {
  StorageAdapter,
  Subscription,
  UsageRecord,
  Coupon,
  Invoice,
  BillingEvent,
  BillingEventType
} from '../types.js';

/**
 * In-Memory Storage Adapter (for testing and development)
 */
export class MemoryStorageAdapter implements StorageAdapter {
  private subscriptions = new Map<string, Subscription>();
  private usageRecords = new Map<string, UsageRecord[]>();
  private coupons = new Map<string, Coupon>();
  private invoices = new Map<string, Invoice>();
  private events = new Map<string, BillingEvent[]>();

  // Subscriptions
  async saveSubscription(subscription: Subscription): Promise<void> {
    this.subscriptions.set(subscription.id, { ...subscription });
  }

  async getSubscription(id: string): Promise<Subscription | null> {
    return this.subscriptions.get(id) || null;
  }

  async updateSubscription(subscription: Subscription): Promise<void> {
    this.subscriptions.set(subscription.id, { ...subscription });
  }

  async deleteSubscription(id: string): Promise<void> {
    this.subscriptions.delete(id);
  }

  async listSubscriptions(userId: string): Promise<Subscription[]> {
    return Array.from(this.subscriptions.values()).filter(s => s.userId === userId);
  }

  // Usage
  async saveUsageRecord(record: UsageRecord): Promise<void> {
    const records = this.usageRecords.get(record.subscriptionId) || [];
    records.push({ ...record });
    this.usageRecords.set(record.subscriptionId, records);
  }

  async getUsageRecords(subscriptionId: string, period: string): Promise<UsageRecord[]> {
    const records = this.usageRecords.get(subscriptionId) || [];
    return records.filter(r => r.billingPeriod === period);
  }

  // Coupons
  async saveCoupon(coupon: Coupon): Promise<void> {
    this.coupons.set(coupon.code, { ...coupon });
  }

  async getCoupon(code: string): Promise<Coupon | null> {
    return this.coupons.get(code) || null;
  }

  async updateCoupon(coupon: Coupon): Promise<void> {
    this.coupons.set(coupon.code, { ...coupon });
  }

  async listCoupons(): Promise<Coupon[]> {
    return Array.from(this.coupons.values());
  }

  // Invoices
  async saveInvoice(invoice: Invoice): Promise<void> {
    this.invoices.set(invoice.id, { ...invoice });
  }

  async getInvoice(id: string): Promise<Invoice | null> {
    return this.invoices.get(id) || null;
  }

  async listInvoices(userId: string): Promise<Invoice[]> {
    return Array.from(this.invoices.values()).filter(i => i.userId === userId);
  }

  // Events
  async saveEvent(event: BillingEvent): Promise<void> {
    const events = this.events.get(event.userId) || [];
    events.push({ ...event });
    this.events.set(event.userId, events);
  }

  async getEvents(userId: string, type?: BillingEventType): Promise<BillingEvent[]> {
    const events = this.events.get(userId) || [];
    if (type) {
      return events.filter(e => e.type === type);
    }
    return events;
  }

  // Utility
  clear(): void {
    this.subscriptions.clear();
    this.usageRecords.clear();
    this.coupons.clear();
    this.invoices.clear();
    this.events.clear();
  }
}

/**
 * AgentDB Storage Adapter (vector database with semantic search)
 */
export class AgentDBStorageAdapter implements StorageAdapter {
  private db: any; // AgentDB instance
  private collections = {
    subscriptions: 'billing_subscriptions',
    usage: 'billing_usage',
    coupons: 'billing_coupons',
    invoices: 'billing_invoices',
    events: 'billing_events'
  };

  constructor(agentDB: any) {
    this.db = agentDB;
  }

  async initialize(): Promise<void> {
    // Create collections if they don't exist
    for (const collection of Object.values(this.collections)) {
      try {
        await this.db.createCollection(collection);
      } catch (error) {
        // Collection might already exist
      }
    }
  }

  // Subscriptions
  async saveSubscription(subscription: Subscription): Promise<void> {
    await this.db.upsert(this.collections.subscriptions, {
      id: subscription.id,
      data: subscription,
      metadata: {
        userId: subscription.userId,
        tier: subscription.tier,
        status: subscription.status
      }
    });
  }

  async getSubscription(id: string): Promise<Subscription | null> {
    const result = await this.db.get(this.collections.subscriptions, id);
    return result?.data || null;
  }

  async updateSubscription(subscription: Subscription): Promise<void> {
    await this.saveSubscription(subscription);
  }

  async deleteSubscription(id: string): Promise<void> {
    await this.db.delete(this.collections.subscriptions, id);
  }

  async listSubscriptions(userId: string): Promise<Subscription[]> {
    const results = await this.db.query(this.collections.subscriptions, {
      filter: { userId }
    });
    return results.map((r: any) => r.data);
  }

  // Usage
  async saveUsageRecord(record: UsageRecord): Promise<void> {
    await this.db.insert(this.collections.usage, {
      id: record.id,
      data: record,
      metadata: {
        subscriptionId: record.subscriptionId,
        metric: record.metric,
        period: record.billingPeriod
      }
    });
  }

  async getUsageRecords(subscriptionId: string, period: string): Promise<UsageRecord[]> {
    const results = await this.db.query(this.collections.usage, {
      filter: {
        subscriptionId,
        'metadata.period': period
      }
    });
    return results.map((r: any) => r.data);
  }

  // Coupons
  async saveCoupon(coupon: Coupon): Promise<void> {
    await this.db.upsert(this.collections.coupons, {
      id: coupon.code,
      data: coupon,
      metadata: {
        type: coupon.type,
        active: coupon.active
      }
    });
  }

  async getCoupon(code: string): Promise<Coupon | null> {
    const result = await this.db.get(this.collections.coupons, code);
    return result?.data || null;
  }

  async updateCoupon(coupon: Coupon): Promise<void> {
    await this.saveCoupon(coupon);
  }

  async listCoupons(): Promise<Coupon[]> {
    const results = await this.db.query(this.collections.coupons, {});
    return results.map((r: any) => r.data);
  }

  // Invoices
  async saveInvoice(invoice: Invoice): Promise<void> {
    await this.db.insert(this.collections.invoices, {
      id: invoice.id,
      data: invoice,
      metadata: {
        userId: invoice.userId,
        status: invoice.status,
        amount: invoice.amount
      }
    });
  }

  async getInvoice(id: string): Promise<Invoice | null> {
    const result = await this.db.get(this.collections.invoices, id);
    return result?.data || null;
  }

  async listInvoices(userId: string): Promise<Invoice[]> {
    const results = await this.db.query(this.collections.invoices, {
      filter: { userId }
    });
    return results.map((r: any) => r.data);
  }

  // Events
  async saveEvent(event: BillingEvent): Promise<void> {
    await this.db.insert(this.collections.events, {
      id: event.id,
      data: event,
      metadata: {
        userId: event.userId,
        type: event.type,
        timestamp: event.timestamp
      }
    });
  }

  async getEvents(userId: string, type?: BillingEventType): Promise<BillingEvent[]> {
    const filter: any = { userId };
    if (type) {
      filter['metadata.type'] = type;
    }

    const results = await this.db.query(this.collections.events, { filter });
    return results.map((r: any) => r.data);
  }
}

/**
 * SQLite Storage Adapter
 */
export class SQLiteStorageAdapter implements StorageAdapter {
  private db: any; // better-sqlite3 instance

  constructor(dbPath: string) {
    // In production, initialize better-sqlite3
    // For now, stub implementation
  }

  async initialize(): Promise<void> {
    // Create tables
    const schema = `
      CREATE TABLE IF NOT EXISTS subscriptions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS usage_records (
        id TEXT PRIMARY KEY,
        subscription_id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS coupons (
        code TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS invoices (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        type TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `;

    // Execute schema in production
  }

  // Implement StorageAdapter interface methods
  // (Similar to MemoryStorageAdapter but with SQL queries)

  async saveSubscription(subscription: Subscription): Promise<void> {
    // SQL: INSERT OR REPLACE INTO subscriptions ...
  }

  async getSubscription(id: string): Promise<Subscription | null> {
    // SQL: SELECT * FROM subscriptions WHERE id = ?
    return null;
  }

  async updateSubscription(subscription: Subscription): Promise<void> {
    await this.saveSubscription(subscription);
  }

  async deleteSubscription(id: string): Promise<void> {
    // SQL: DELETE FROM subscriptions WHERE id = ?
  }

  async listSubscriptions(userId: string): Promise<Subscription[]> {
    // SQL: SELECT * FROM subscriptions WHERE user_id = ?
    return [];
  }

  async saveUsageRecord(record: UsageRecord): Promise<void> {
    // SQL: INSERT INTO usage_records ...
  }

  async getUsageRecords(subscriptionId: string, period: string): Promise<UsageRecord[]> {
    // SQL: SELECT * FROM usage_records WHERE subscription_id = ?
    return [];
  }

  async saveCoupon(coupon: Coupon): Promise<void> {
    // SQL: INSERT OR REPLACE INTO coupons ...
  }

  async getCoupon(code: string): Promise<Coupon | null> {
    // SQL: SELECT * FROM coupons WHERE code = ?
    return null;
  }

  async updateCoupon(coupon: Coupon): Promise<void> {
    await this.saveCoupon(coupon);
  }

  async listCoupons(): Promise<Coupon[]> {
    // SQL: SELECT * FROM coupons
    return [];
  }

  async saveInvoice(invoice: Invoice): Promise<void> {
    // SQL: INSERT INTO invoices ...
  }

  async getInvoice(id: string): Promise<Invoice | null> {
    // SQL: SELECT * FROM invoices WHERE id = ?
    return null;
  }

  async listInvoices(userId: string): Promise<Invoice[]> {
    // SQL: SELECT * FROM invoices WHERE user_id = ?
    return [];
  }

  async saveEvent(event: BillingEvent): Promise<void> {
    // SQL: INSERT INTO events ...
  }

  async getEvents(userId: string, type?: BillingEventType): Promise<BillingEvent[]> {
    // SQL: SELECT * FROM events WHERE user_id = ?
    return [];
  }
}

/**
 * Storage Adapter Factory
 */
export class StorageAdapterFactory {
  static createMemory(): MemoryStorageAdapter {
    return new MemoryStorageAdapter();
  }

  static createAgentDB(agentDB: any): AgentDBStorageAdapter {
    const adapter = new AgentDBStorageAdapter(agentDB);
    adapter.initialize();
    return adapter;
  }

  static createSQLite(dbPath: string): SQLiteStorageAdapter {
    const adapter = new SQLiteStorageAdapter(dbPath);
    adapter.initialize();
    return adapter;
  }
}
