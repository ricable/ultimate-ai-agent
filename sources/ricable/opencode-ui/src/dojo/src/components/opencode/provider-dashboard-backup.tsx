/**
 * Provider Dashboard - Simplified working version
 */

"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export const ProviderDashboard = () => {
  return (
    <div className="p-6">
      <Card>
        <CardHeader>
          <CardTitle>Provider Dashboard</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Provider dashboard is loading...</p>
          <Button>Test Button</Button>
        </CardContent>
      </Card>
    </div>
  );
};