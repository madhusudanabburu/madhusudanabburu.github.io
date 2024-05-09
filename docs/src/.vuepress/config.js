const { description } = require('../../package')

module.exports = {
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: 'Documentation',
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#description
   */
  description: description,

  /**
   * Extra tags to be injected to the page HTML `<head>`
   *
   * ref：https://v1.vuepress.vuejs.org/config/#head
   */
  head: [
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }]
  ],

  /**
   * Theme configuration, here is the default theme configuration for VuePress.
   *
   * ref：https://v1.vuepress.vuejs.org/theme/default-theme-config.html
   */
  themeConfig: {
    repo: '',
    editLinks: false,
    docsDir: '',
    editLinkText: '',
    lastUpdated: false,
    nav: [
      {
        text: 'ML Guide',
        link: '/guide/',
      },
      {
        text: 'Home',
        link: '/',
      },
    ],
    sidebar: [
      {
        title: 'Prerequisites',
        path: '/guide/',
        collapsable: false,
        children: [
          '/guide/',
          '/guide/kubeflow'
        ]
      },
      {
        title: 'Notebooks',
        path: '/guide/notebooks/',
        collapsable: false,
        children: [
          '/guide/notebooks/',
          '/guide/notebooks/twitter',
          '/guide/notebooks/covid'
        ]      
      },
      {
        title: 'Kubeflow Pipelines',
        path: '/guide/pipelines/',
        collapsable: false,
        children: [
          '/guide/pipelines/',
          '/guide/pipelines/twitter_supervised'
        ]
      },
      {
        title: 'Model Deployment',
        path: '/guide/vertexai/',
        collapsable: false,
        children: [
          '/guide/vertexai/'
        ]
      }
    ]
  },

  /**
   * Apply plugins，ref：https://v1.vuepress.vuejs.org/zh/plugin/
   */
  plugins: [
    ['@vuepress/plugin-google-analytics',
      {
        ga: 'G-GNZH16V1S9' // Measurement ID
      }
    ]
  ],

  head: [
    [
        'script',
        {
            async: true,
            src: 'https://www.googletagmanager.com/gtag/js?id=G-GNZH16V1S9',
        },
    ],
    [
        'script',
        {},
        [
            "window.dataLayer = window.dataLayer || [];\nfunction gtag(){dataLayer.push(arguments);}\ngtag('js', new Date());\ngtag('config', 'G-GNZH16V1S9');",
        ],
    ],
],
}
