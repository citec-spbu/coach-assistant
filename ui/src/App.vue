<template>
  <div id="app">
    <component :is="pages[currentPage]" @navigate="handleNavigation" />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import HomePage from './components/HomePage.vue'
import UploadPage from './components/UploadPage.vue'

const pages = {
  home: HomePage,
  upload: UploadPage
}

const currentPage = ref('home')

const handleNavigation = (page) => {
  console.log('Навигация на:', page)
  currentPage.value = page
}
</script>