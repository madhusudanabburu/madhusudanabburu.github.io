
import Image from 'next/image'
 
const UnoptimizedImage = (props) => {
  return <Image {...props} unoptimized />
}

const withNextra = require('nextra')({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.tsx',
})

module.exports = {withNextra, UnoptimizedImage}
