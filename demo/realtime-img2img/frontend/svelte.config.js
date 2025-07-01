import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const config = {
  extensions: ['.svelte'],
  compilerOptions: {},
  preprocess: vitePreprocess({ postcss: true }),
  kit: {
    adapter: adapter({
      pages: 'public',
      assets: 'public',
      fallback: undefined,
      precompress: false,
      strict: true
    })
  },
  vitePlugin: {
    exclude: [],
    // experimental options
    experimental: {}
  }
};

export default config;