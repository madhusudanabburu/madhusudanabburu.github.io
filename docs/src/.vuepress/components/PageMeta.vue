<template>
  <div class="page-meta" v-if="hasDate">
    <span v-if="createdDate">
      Created: {{ formatDate(createdDate) }}
    </span>
    <span v-if="updatedDate">
      Updated: {{ formatDate(updatedDate) }}
    </span>
  </div>
</template>

<script>
export default {
  computed: {
    createdDate() {
      if (this.$page.frontmatter.date) {
        return this.$page.frontmatter.date;
      }
      return this.$page.createdTime;
    },
    updatedDate() {
      return this.$page.lastUpdated;
    },
    hasDate() {
      return this.createdDate || this.updatedDate;
    }
  },
  methods: {
    formatDate(date) {
      if (!date) return '';
      
      const d = new Date(date);
      
      // "January 15, 2024" format
      return d.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
      });
    }
  }
}
</script>

<style scoped>
.page-meta {
  margin-top: 2rem;
  padding-top: 1rem;
  color: #999;
  font-size: 0.9rem;
  text-align: right;
}
.page-meta span:not(:last-child) {
  margin-right: 1rem;
}
</style>