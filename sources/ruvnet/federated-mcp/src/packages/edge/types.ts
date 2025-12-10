// Intent Detection Types
export interface IntentDetectionPayload {
  meetingId: string;
  transcriptionText: string;
  participants: string[];
  metadata: {
    duration: number;
    date: string;
  }
}

export interface DetectedIntent {
  type: string;
  confidence: number;
  segment: {
    text: string;
    timestamp: number;
    speaker: string;
  }
  metadata: {
    context: string[];
    entities: string[];
  }
}

// Meeting Info Types
export interface TranscriptSummary {
  keywords: string[];
  action_items: string[];
  outline: string[];
  shorthand_bullet: string[];
  overview: string;
  bullet_gist: string[];
  gist: string;
  short_summary: string;
}

export interface TranscriptResponse {
  transcript?: {
    summary: TranscriptSummary;
  };
}

// Webhook Types
export interface WebhookPayload {
  meetingId: string;
  eventType: string;
  clientReferenceId?: string;
}

// Shared Response Type
export interface EdgeResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
}
