const { description } = require('../../package')

module.exports = {
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: 'Tutorials',
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
  markdown: {
    extendMarkdown: md => {
      md.use(require('markdown-it-html5-embed'), {
        html5embed: {
          useImageSyntax: true,
          useLinkSyntax: false
        }
      })
    }
  },
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
        title: 'Introduction',
        path: '/guide/',
        collapsable: false,
        children: [
          '/guide/'
        ]
      },
      {
        title: 'Prerequisites',
        path: '/guide/prerequisites/',
        collapsable: false,
        children: [
          '/guide/prerequisites/',
          '/guide/prerequisites/kubeflow',
          '/guide/prerequisites/minio'
       ]
      },
      {
        title: 'Predictive AI - Jupyter Notebooks',
        path: '/guide/notebooks/',
        collapsable: false,
        children: [
          '/guide/notebooks/',
          '/guide/notebooks/twitter',
          '/guide/notebooks/covid'
        ]      
      },
      {
        title: 'Predictive AI - Kubeflow Pipelines',
        path: '/guide/pipelines/',
        collapsable: false,
        children: [
          '/guide/pipelines/',
          '/guide/pipelines/twitter_supervised',
          '/guide/pipelines/covid'
        ]
      },
      {
        title: 'Opensource LLMs',
        path: '/guide/ossllm/',
        collapsable: false,
        children: [
          '/guide/ossllm/llama3' 
        ]
      },
      {
        title: 'Generative AI - Applications',
        path: '/guide/genaiapps/',
        collapsable: false,
        children: [
          '/guide/genaiapps/ragchatbotwithsite',
          '/guide/genaiapps/ragchatbotwithdocs',
          '/guide/genaiapps/aiagent'
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
    ],
    ['markdown-it-html5-embed',
      {
        attributes: {
          'audio': 'width="120" controls class="audioplayer"',
          'video': 'width="120" height="140" class="videoplayer" controls'
        }
      }

    ],
    [require('./plugins/creation-date.js')] 
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
