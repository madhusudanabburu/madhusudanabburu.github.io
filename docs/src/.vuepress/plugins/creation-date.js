const { execSync } = require('child_process');

module.exports = (options = {}, context) => ({
  extendPageData($page) {
    const { _filePath } = $page;
    if (!_filePath) return;

    try {
      // Get the first commit date for this file from git
      const gitDate = execSync(
        `git log --follow --format=%aI --reverse -- "${_filePath}" | head -1`,
        { encoding: 'utf-8', cwd: process.cwd() }
      ).trim();
      
      if (gitDate) {
        const creationDate = new Date(gitDate);
        $page.createdTime = creationDate.getTime();
        $page.createdDate = creationDate.toISOString();
      }
    } catch (error) {
      // Fallback to filesystem date (for local dev)
      try {
        const fs = require('fs');
        const stats = fs.statSync(_filePath);
        $page.createdTime = stats.birthtime.getTime();
        $page.createdDate = stats.birthtime.toISOString();
      } catch (e) {
        // Silent fail
      }
    }
  }
});