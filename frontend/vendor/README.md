# Vendored front-end libraries

Served locally so the app works offline (e.g. on hospital networks), versions are
pinned, and the Content-Security-Policy can forbid third-party scripts.

| Library | Version | Files | License |
|---|---|---|---|
| [Chart.js](https://www.chartjs.org) | 4.4.6 | `chart.umd.min.js` | MIT (`chart.js.LICENSE.md`) |
| [Font Awesome Free](https://fontawesome.com) | 6.4.0 | `fontawesome/css/{fontawesome,solid}.min.css`, `fontawesome/webfonts/fa-solid-900.woff2` | Icons CC BY 4.0, fonts SIL OFL 1.1, code MIT (`fontawesome/LICENSE.txt`) |

Both were taken unmodified from the official npm packages (`npm pack chart.js@4.4.6`,
`npm pack @fortawesome/fontawesome-free@6.4.0`).
