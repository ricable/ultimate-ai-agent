import { create, verify } from "https://deno.land/x/djwt/mod.ts";

export class AuthManager {
  private key: Promise<CryptoKey>;

  constructor(secret: string) {
    this.key = this.initializeKey(secret);
  }

  private async initializeKey(secret: string): Promise<CryptoKey> {
    const encoder = new TextEncoder();
    const keyData = encoder.encode(secret);
    return await crypto.subtle.importKey(
      "raw",
      keyData,
      { name: "HMAC", hash: "SHA-512" },
      false,
      ["sign", "verify"]
    );
  }

  async createToken(payload: Record<string, unknown>): Promise<string> {
    const key = await this.key;
    return await create({ alg: "HS512", typ: "JWT" }, payload, key);
  }

  async verifyToken(token: string): Promise<Record<string, unknown>> {
    const key = await this.key;
    return await verify(token, key);
  }
}
