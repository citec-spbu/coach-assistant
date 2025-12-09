<template>
  <div class="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
    <!-- Navigation -->
    <nav class="flex justify-between items-center p-6">
      <div class="flex items-center space-x-2">
        <div class="w-10 h-10 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg flex items-center justify-center">
          <Music class="w-6 h-6 text-white" />
        </div>
        <span class="text-white text-xl font-bold">Ассистент тренера</span>
      </div>
      <button
        @click="$emit('navigate', 'upload')"
        class="bg-gradient-to-r from-pink-500 to-violet-500 text-white px-6 py-2 rounded-full font-medium hover:from-pink-600 hover:to-violet-600 transition-all duration-300 flex items-center space-x-2"
      >
        <span>Начать анализ</span>
        <ChevronRight class="w-4 h-4" />
      </button>
    </nav>

    <!-- Hero -->
    <div class="container mx-auto px-6 py-16 text-center">
      <h1 class="text-5xl md:text-7xl font-bold text-white mb-6">
        Анализируйте ваш танец
        <span class="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-violet-400">как профессионал</span>
      </h1>
      <p class="text-xl text-gray-200 mb-12 max-w-3xl mx-auto">
        Наш ИИ сравнивает ваше исполнение с эталонными движениями и дает подробную обратную связь для улучшения вашей техники
      </p>

      <div class="flex flex-col sm:flex-row gap-4 justify-center mb-16">
        <button
          @click="$emit('navigate', 'upload')"
          class="bg-gradient-to-r from-pink-500 to-violet-500 text-white px-8 py-4 rounded-full text-lg font-semibold hover:from-pink-600 hover:to-violet-600 transition-all duration-300 transform hover:scale-105"
        >
          Загрузить видео
        </button>
        <button class="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-full text-lg font-semibold border border-white/20 hover:bg-white/20 transition-all duration-300">
          Посмотреть демо
        </button>
      </div>

      <!-- Features -->
      <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <div class="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <div class="w-12 h-12 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg flex items-center justify-center mb-4 mx-auto">
            <Video class="w-6 h-6 text-white" />
          </div>
          <h3 class="text-xl font-semibold text-white mb-2">Видеоанализ</h3>
          <p class="text-gray-200">Точное отслеживание движений с помощью компьютерного зрения</p>
        </div>
        <div class="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <div class="w-12 h-12 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg flex items-center justify-center mb-4 mx-auto">
            <TrendingUp class="w-6 h-6 text-white" />
          </div>
          <h3 class="text-xl font-semibold text-white mb-2">Обратная связь</h3>
          <p class="text-gray-200">Детальные рекомендации по улучшению техники и синхронизации</p>
        </div>
        <div class="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
          <div class="w-12 h-12 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg flex items-center justify-center mb-4 mx-auto">
            <Users class="w-6 h-6 text-white" />
          </div>
          <h3 class="text-xl font-semibold text-white mb-2">Сравнение</h3>
          <p class="text-gray-200">Сравнение с профессиональными танцорами и эталонными движениями</p>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer class="text-center text-gray-300 py-8">
      <p>Учебная практика. Проект "Ассистент тренера"</p>
    </footer>
  </div>
</template>

<script setup>
import { Music, ChevronRight, Video, TrendingUp, Users } from 'lucide-vue-next'

defineEmits(['navigate'])
</script>