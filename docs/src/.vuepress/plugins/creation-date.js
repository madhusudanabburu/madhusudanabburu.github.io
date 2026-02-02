const fs = require('fs');

module.exports = (options = {}, context) => ({
  extendPageData($page) {
    const { _filePath } = $page;
    if (!_filePath) return;

    try {
      // _filePath is already an absolute path, use it directly
      const stats = fs.statSync(_filePath);
      
      $page.createdTime = stats.birthtime.getTime();
      $page.createdDate = stats.birthtime.toISOString();
      
      console.log('✓ Set date for:', _filePath, '→', stats.birthtime);
    } catch (error) {
      console.error('✗ Error for:', _filePath, error.message);
    }
  }
});