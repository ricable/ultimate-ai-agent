/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Don't attempt to import node-specific modules on the client side
      config.resolve.fallback = {
        ...config.resolve.fallback,
        child_process: false,
        fs: false,
        net: false,
        tls: false
      }
    }
    return config
  }
}

export default nextConfig
