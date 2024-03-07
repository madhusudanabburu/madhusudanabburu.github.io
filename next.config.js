const withMarkdoc = require('@markdoc/next.js');

module.exports =
  withMarkdoc({schemaPath: './markdoc', mode: 'static'})({
    pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdoc'],
  });

/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  output: 'export',

  assetPrefix: ".",

  basePath: '/out',
 
  // Optional: Change links `/me` -> `/me/` and emit `/me.html` -> `/me/index.html`
  trailingSlash: true,
 
  // Optional: Prevent automatic `/me` -> `/me/`, instead preserve `href`
  skipTrailingSlashRedirect: true,
 
  // Optional: Change the output directory `out` -> `dist`
  // distDir: 'dist',
}
 
/*module.exports = nextConfig*/