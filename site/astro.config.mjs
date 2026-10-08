import { defineConfig } from "astro/config";
// Static export only (GitHub Pages / Cloudflare Pages). SITE / BASE come from the deploy step.
export default defineConfig({
  output: "static",
  site: process.env.SITE_URL || "http://localhost:4321",
  base: process.env.SITE_BASE || "/",
  trailingSlash: "ignore",
});
