const withNextra = require('nextra')({
  theme: 'madhusudanabburu.github.io',
  themeConfig: './theme.config.tsx',
})

module.exports = {withNextra, images: {
  unoptimized: true,
}}
