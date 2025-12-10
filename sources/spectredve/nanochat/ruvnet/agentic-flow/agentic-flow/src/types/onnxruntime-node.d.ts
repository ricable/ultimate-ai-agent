// Stub types for optional dependency onnxruntime-node
declare module 'onnxruntime-node' {
  export namespace Tensor {
    const type: any;
  }
  export class InferenceSession {
    static create(path: string, options?: any): Promise<any>;
    run(feeds: any, options?: any): Promise<any>;
    [key: string]: any;
  }
  export const env: any;
}
