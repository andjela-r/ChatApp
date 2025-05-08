import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      config.watchOptions = {
        poll: 1000,            // check for changes every second
        aggregateTimeout: 300, // delay rebuild slightly to batch rapid changes
        ignored: /node_modules/, // improve performance
      };
    }
    return config;
  },
};

export default nextConfig;
