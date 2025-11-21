module.exports = {
  launch: {
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--disable-gpu'
    ],
    defaultViewport: {
      width: 1280,
      height: 800
    }
  },
  server: {
    command: 'echo "Application should already be running on localhost:5173"',
    port: 5173,
    launchTimeout: 5000,
    debug: false
  }
};