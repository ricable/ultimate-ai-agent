/**
 * Market Integration
 * 
 * Handles market operations for Claude capacity sharing
 * Integrates with Synaptic Neural Mesh market infrastructure
 */

import { v4 as uuidv4 } from 'uuid';
import WebSocket from 'ws';

export class MarketIntegration {
  constructor() {
    this.activeOffers = new Map();
    this.activeBids = new Map();
    this.marketHistory = [];
    this.wsConnection = null;
    this.marketEndpoint = process.env.SYNAPTIC_MARKET_ENDPOINT || 'ws://localhost:8080/market';
    this.nodeId = this.generateNodeId();
  }

  /**
   * Advertise available Claude capacity
   */
  async advertise(options = {}) {
    const offer = {
      id: uuidv4(),
      nodeId: this.nodeId,
      type: 'claude_capacity',
      slots: options.slots || 1,
      price: options.price || 5,
      capabilities: options.capabilities || ['claude-3-sonnet'],
      timestamp: new Date().toISOString(),
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // 24 hours
      compliance: options.compliance || {},
      metadata: {
        dockerImage: 'synaptic-mesh/claude-max:latest',
        maxTokens: 1000,
        timeout: 300,
        encryption: true
      }
    };

    this.activeOffers.set(offer.id, offer);
    
    // Broadcast offer to market
    await this.broadcastToMarket({
      type: 'offer',
      data: offer
    });

    console.log(`📢 Advertised capacity: ${offer.slots} slots at ${offer.price} RUV each`);
    return offer;
  }

  /**
   * Place bid for Claude task execution
   */
  async placeBid(options = {}) {
    if (!options.taskId) {
      throw new Error('Task ID required for bidding');
    }

    const bid = {
      id: uuidv4(),
      nodeId: this.nodeId,
      taskId: options.taskId,
      maxPrice: options.maxPrice || 10,
      capabilities: options.capabilities || ['claude-3-sonnet'],
      timestamp: new Date().toISOString(),
      expires: new Date(Date.now() + 30 * 60 * 1000).toISOString(), // 30 minutes
      requirements: {
        maxTokens: options.maxTokens || 1000,
        timeout: options.timeout || 300,
        model: options.model || 'claude-3-sonnet-20240229'
      }
    };

    this.activeBids.set(bid.id, bid);

    // Submit bid to market
    await this.broadcastToMarket({
      type: 'bid',
      data: bid
    });

    console.log(`💰 Placed bid: ${bid.maxPrice} RUV for task ${bid.taskId}`);
    return bid;
  }

  /**
   * Accept a bid for task execution
   */
  async acceptBid(bidId) {
    const acceptance = {
      id: uuidv4(),
      nodeId: this.nodeId,
      bidId: bidId,
      timestamp: new Date().toISOString(),
      status: 'accepted'
    };

    await this.broadcastToMarket({
      type: 'accept',
      data: acceptance
    });

    console.log(`✅ Accepted bid: ${bidId}`);
    return acceptance;
  }

  /**
   * Complete task and settle payment
   */
  async settleTask(bidId, result) {
    const settlement = {
      id: uuidv4(),
      nodeId: this.nodeId,
      bidId: bidId,
      timestamp: new Date().toISOString(),
      result: {
        success: result.success,
        tokens: result.tokens,
        executionTime: result.executionTime,
        hash: this.generateResultHash(result)
      }
    };

    await this.broadcastToMarket({
      type: 'settle',
      data: settlement
    });

    // Remove completed bid
    this.activeBids.delete(bidId);

    console.log(`💳 Settled task: ${bidId}`);
    return settlement;
  }

  /**
   * Get market statistics
   */
  async getMarketStats() {
    return {
      activeOffers: this.activeOffers.size,
      activeBids: this.activeBids.size,
      completedTasks: this.marketHistory.filter(h => h.type === 'settle').length,
      averagePrice: this.calculateAveragePrice(),
      nodeReputation: await this.getNodeReputation(),
      marketActivity: this.getRecentActivity()
    };
  }

  /**
   * Cancel offer or bid
   */
  async cancel(id, type) {
    let cancelled = null;

    if (type === 'offer' && this.activeOffers.has(id)) {
      cancelled = this.activeOffers.get(id);
      this.activeOffers.delete(id);
    } else if (type === 'bid' && this.activeBids.has(id)) {
      cancelled = this.activeBids.get(id);
      this.activeBids.delete(id);
    }

    if (cancelled) {
      await this.broadcastToMarket({
        type: 'cancel',
        data: {
          id: uuidv4(),
          nodeId: this.nodeId,
          targetId: id,
          targetType: type,
          timestamp: new Date().toISOString()
        }
      });

      console.log(`❌ Cancelled ${type}: ${id}`);
    }

    return cancelled;
  }

  /**
   * Get node reputation score
   */
  async getNodeReputation() {
    const completedTasks = this.marketHistory.filter(h => 
      h.type === 'settle' && h.data.nodeId === this.nodeId
    );

    if (completedTasks.length === 0) {
      return { score: 100, tasks: 0, rating: 'New' };
    }

    const successfulTasks = completedTasks.filter(t => t.data.result.success);
    const successRate = (successfulTasks.length / completedTasks.length) * 100;

    let rating = 'Poor';
    if (successRate >= 95) rating = 'Excellent';
    else if (successRate >= 85) rating = 'Good';
    else if (successRate >= 70) rating = 'Fair';

    return {
      score: Math.round(successRate),
      tasks: completedTasks.length,
      rating,
      successRate
    };
  }

  /**
   * Start market connection
   */
  async connect() {
    try {
      this.wsConnection = new WebSocket(this.marketEndpoint);

      this.wsConnection.on('open', () => {
        console.log('🔗 Connected to Synaptic Market');
        this.sendHeartbeat();
      });

      this.wsConnection.on('message', (data) => {
        this.handleMarketMessage(JSON.parse(data.toString()));
      });

      this.wsConnection.on('close', () => {
        console.log('🔌 Disconnected from Synaptic Market');
        // Attempt reconnection
        setTimeout(() => this.connect(), 5000);
      });

      this.wsConnection.on('error', (error) => {
        console.warn('Market connection error:', error.message);
      });

    } catch (error) {
      console.warn('Failed to connect to market:', error.message);
      // Continue without market connection (offline mode)
    }
  }

  /**
   * Disconnect from market
   */
  async disconnect() {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Get recent market activity
   */
  getRecentActivity(hours = 24) {
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    return this.marketHistory.filter(h => new Date(h.timestamp) >= since);
  }

  /**
   * Search market for available capacity
   */
  async searchMarket(requirements = {}) {
    // In a real implementation, this would query the market
    // For now, return available offers that match requirements
    const matches = Array.from(this.activeOffers.values()).filter(offer => {
      if (requirements.maxPrice && offer.price > requirements.maxPrice) return false;
      if (requirements.capabilities && !requirements.capabilities.some(cap => offer.capabilities.includes(cap))) return false;
      if (requirements.minSlots && offer.slots < requirements.minSlots) return false;
      
      return true;
    });

    return matches.sort((a, b) => a.price - b.price); // Sort by price
  }

  // Private helper methods

  generateNodeId() {
    // Generate unique node ID based on machine characteristics
    const nodeInfo = {
      platform: process.platform,
      arch: process.arch,
      pid: process.pid,
      timestamp: Date.now()
    };
    
    const hash = require('crypto')
      .createHash('sha256')
      .update(JSON.stringify(nodeInfo))
      .digest('hex');
    
    return `node_${hash.substring(0, 16)}`;
  }

  generateResultHash(result) {
    return require('crypto')
      .createHash('sha256')
      .update(JSON.stringify({
        success: result.success,
        tokens: result.tokens,
        timestamp: result.timestamp
      }))
      .digest('hex');
  }

  async broadcastToMarket(message) {
    this.marketHistory.push({
      ...message,
      timestamp: new Date().toISOString()
    });

    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      try {
        this.wsConnection.send(JSON.stringify(message));
      } catch (error) {
        console.warn('Failed to broadcast to market:', error.message);
      }
    } else {
      console.log('📤 Market message queued (offline):', message.type);
    }
  }

  handleMarketMessage(message) {
    console.log('📨 Market message received:', message.type);

    switch (message.type) {
      case 'bid':
        this.handleIncomingBid(message.data);
        break;
      case 'accept':
        this.handleBidAcceptance(message.data);
        break;
      case 'settle':
        this.handleTaskSettlement(message.data);
        break;
      case 'cancel':
        this.handleCancellation(message.data);
        break;
      case 'heartbeat':
        this.handleHeartbeat(message.data);
        break;
      default:
        console.log('Unknown market message type:', message.type);
    }
  }

  handleIncomingBid(bid) {
    // Check if we have matching capacity
    const matchingOffers = Array.from(this.activeOffers.values()).filter(offer => 
      offer.price <= bid.maxPrice &&
      bid.capabilities.some(cap => offer.capabilities.includes(cap))
    );

    if (matchingOffers.length > 0) {
      console.log(`💡 Received matching bid: ${bid.id} (${bid.maxPrice} RUV)`);
      // Auto-accept if configured, otherwise require manual approval
    }
  }

  handleBidAcceptance(acceptance) {
    if (this.activeBids.has(acceptance.bidId)) {
      console.log(`🎉 Bid accepted: ${acceptance.bidId}`);
      // Start task execution
    }
  }

  handleTaskSettlement(settlement) {
    console.log(`💰 Task settled: ${settlement.bidId}`);
    // Process payment
  }

  handleCancellation(cancellation) {
    console.log(`❌ Market cancellation: ${cancellation.targetType} ${cancellation.targetId}`);
  }

  handleHeartbeat(data) {
    // Market is alive, update connection status
  }

  sendHeartbeat() {
    if (this.wsConnection && this.wsConnection.readyState === WebSocket.OPEN) {
      this.wsConnection.send(JSON.stringify({
        type: 'heartbeat',
        nodeId: this.nodeId,
        timestamp: new Date().toISOString()
      }));
    }
    
    // Schedule next heartbeat
    setTimeout(() => this.sendHeartbeat(), 30000); // 30 seconds
  }

  calculateAveragePrice() {
    const settlements = this.marketHistory.filter(h => h.type === 'settle');
    if (settlements.length === 0) return 0;

    const offers = this.marketHistory.filter(h => h.type === 'offer');
    const totalPrice = offers.reduce((sum, offer) => sum + offer.data.price, 0);
    
    return totalPrice / offers.length;
  }

  /**
   * Export market data for analysis
   */
  async exportMarketData() {
    return {
      nodeId: this.nodeId,
      activeOffers: Array.from(this.activeOffers.values()),
      activeBids: Array.from(this.activeBids.values()),
      history: this.marketHistory,
      stats: await this.getMarketStats(),
      reputation: await this.getNodeReputation(),
      exported: new Date().toISOString()
    };
  }

  /**
   * Import market data (for backup/restore)
   */
  async importMarketData(data) {
    if (data.nodeId !== this.nodeId) {
      console.warn('Importing data from different node');
    }

    // Restore active offers and bids
    data.activeOffers?.forEach(offer => {
      this.activeOffers.set(offer.id, offer);
    });

    data.activeBids?.forEach(bid => {
      this.activeBids.set(bid.id, bid);
    });

    // Restore history (merge, don't overwrite)
    if (data.history) {
      this.marketHistory = [...this.marketHistory, ...data.history]
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }

    console.log('📥 Market data imported successfully');
  }
}