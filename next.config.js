const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
})
module.exports = {
  images: {
    unoptimized: true,
  },
}
module.exports = withNextra()
