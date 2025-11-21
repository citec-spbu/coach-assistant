<template>
  <div class="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 p-6">
    <!-- Navigation -->
    <nav class="flex justify-between items-center mb-8">
      <button
        @click="$emit('navigate', 'home')"
        class="text-gray-300 hover:text-white transition-colors duration-200 flex items-center space-x-2"
      >
        <ArrowLeft class="w-5 h-5" />
        <span>Назад на главную</span>
      </button>
      <div class="flex items-center space-x-2">
        <div class="w-8 h-8 bg-gradient-to-r from-pink-500 to-violet-500 rounded-lg flex items-center justify-center">
          <Music class="w-4 h-4 text-white" />
        </div>
        <span class="text-white font-semibold">Ассистент тренера</span>
      </div>
    </nav>

    <div class="max-w-4xl mx-auto">
      <h1 class="text-3xl md:text-4xl font-bold text-white mb-8 text-center">
        Загрузка и анализ видео
      </h1>

      <!-- Upload Section -->
      <div class="bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-700">
        <div
          class="border-2 border-dashed border-gray-600 rounded-xl p-12 text-center cursor-pointer hover:border-pink-500 transition-colors duration-200"
          @click="triggerUpload"
          @dragover.prevent
          @drop.prevent="handleDrop"
        >
          <Upload class="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 class="text-xl font-semibold text-white mb-2">Загрузить видео</h3>
          <p class="text-gray-400 mb-4">Нажмите или перетащите видео файл</p>
          <p class="text-sm text-gray-500">Поддерживаемые форматы: MP4, MOV, AVI</p>
          <input
            ref="fileInput"
            type="file"
            accept="video/*"
            @change="handleVideoUpload"
            class="hidden"
          />
        </div>

        <div v-if="uploadedVideo" class="mt-6">
          <h3 class="text-lg font-semibold text-white mb-4">Превью видео</h3>
          <div class="bg-black rounded-lg overflow-hidden">
            <video
              ref="videoPlayer"
              :src="uploadedVideo"
              @timeupdate="onTimeUpdate"
              @loadedmetadata="handleMetadataLoaded"
              class="w-full h-auto max-h-96 object-contain"
              controls
              muted
            ></video>
          </div>
          <div class="text-white font-mono text-center mt-2">
            Текущее время: {{ timeDisplay }}
          </div>
        </div>
      </div>

      <!-- Trim Settings -->
      <div v-if="uploadedVideo" class="bg-gray-800 rounded-2xl p-8 mb-8 border border-gray-700">
        <h3 class="text-xl font-semibold text-white mb-6">Настройки обрезки</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label class="block text-gray-300 mb-2">Начало</label>
            <TimePickerMs v-model="startTime" />
          </div>
          <div>
            <label class="block text-gray-300 mb-2">Конец</label>
            <TimePickerMs v-model="endTime" />
          </div>
        </div>

        <!-- Analyze Button -->
        <div class="text-center mt-8">
          <button
            @click="cutVideo"
            :disabled="isAnalyzing"
            class="bg-gradient-to-r from-pink-500 to-violet-500 text-white px-8 py-4 rounded-full text-lg font-semibold hover:from-pink-600 hover:to-violet-600 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto"
          >
            <template v-if="isAnalyzing">
              <Zap class="w-5 h-5 mr-2 animate-pulse" />
              Анализируем...
            </template>
            <template v-else>
              <Play class="w-5 h-5 mr-2" />
              Отправить видео на анализ
            </template>
          </button>
        </div>
      </div>

      <!-- Processed Video Section -->
      <div v-if="processedVideoUrl" class="bg-gray-800 rounded-2xl p-8 border border-gray-700">
        <h3 class="text-2xl font-bold text-white mb-6 text-center">Результат анализа</h3>
        <!-- Scores -->
        <div class="grid place-items-center mb-8">
          <div class="bg-gradient-to-r from-pink-500/20 to-violet-500/20 rounded-xl p-4 text-center border border-pink-500/30 w-64">
          <div class="text-3xl font-bold text-pink-400">{{ analysisResult.overall }}%</div>
          <div class="text-gray-300">Общий балл</div>
        </div>
      </div>

        <!-- Feedback -->
        <div class="bg-gray-700/50 rounded-xl p-6">
          <h4 class="text-xl font-semibold text-white mb-4">Выделенные позы</h4>
          <ul class="space-y-3">
            <li v-for="(item, idx) in analysisResult.feedback" :key="idx" class="flex items-start space-x-3">
              <div class="w-2 h-2 bg-pink-500 rounded-full mt-2 flex-shrink-0"></div>
              <span class="text-gray-200">{{ item }}</span>
            </li>
          </ul>
        </div>



         <!-- Video Result Preview -->
        <div class="mt-8">
          <h4 class="text-xl font-semibold text-white mb-4">Видео с анализом</h4>
          <div class="bg-black rounded-lg overflow-hidden">
            <video
              ref="processedVideo"
              :src="processedVideoUrl"
              class="w-full h-auto max-h-96 object-contain"
              controls
            ></video>
            <div class="bg-gray-900/80 p-4 text-center">
              <p class="text-gray-300">Видео с наложенными метками анализа (симуляция)</p>
            </div>
          </div>
        </div>
        
        <div class="text-center mt-4">
          <a
            :href="processedVideoUrl"
            download
            class="inline-flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors"
          >
            <Download class="w-5 h-5" />
            Скачать видео
          </a>
        </div>
      </div>

      <!-- Message -->
      <div v-if="message" class="mt-4 p-4 bg-blue-500/20 text-blue-300 rounded-lg text-center">
        {{ message }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { io } from 'socket.io-client'
import axios from 'axios'
import { ArrowLeft, Music, Upload, Zap, Play, Download } from 'lucide-vue-next'
import TimePickerMs from './TimePickerMs.vue'

defineEmits(['navigate'])

const uploadedVideo = ref(null)
const startTime = ref('00:00:00.000')
const endTime = ref('00:00:30.000')
const isAnalyzing = ref(false)
const analysisResult = ref(null)
const fileInput = ref(null)
const processedVideo = ref(null)
const videoPlayer = ref(null)
const timeDisplay = ref('00:00:00.000')
const message = ref('')

let socket = null
const processedVideoUrl = ref(null)
const uploadedFile = ref(null);

onMounted(() => {
  socket = io()
  
  socket.on('video-ready', (data) => {
  console.log('Видео готово:', data);

  if (!data || !data.download_url || data.download_url.trim() === '') {
    message.value = 'Видео от сервера не получено или ссылка отсутствует';
    console.warn('Нет данных или URL обработанного видео');
    isAnalyzing.value = false;
    return;
  }

  if (data.status === "done") {
    const result = JSON.parse(data.metadata);
    analysisResult.value = {
      overall: Math.round(result.confidence * 100),
      feedback: [getFigure(result.figures)]
    };

    processedVideoUrl.value = data.download_url;
    if (processedVideo.value) {
      processedVideo.value.src = processedVideoUrl.value;
      processedVideo.value.load();
      processedVideo.value.play().catch(() => {});
    }

    message.value = 'Видео успешно обработано!';
  } else {
    message.value = "Ошибка обработки видео";
  }

  isAnalyzing.value = false;
});
})

onUnmounted(() => {
  if (socket) {
    socket.disconnect()
  }
})


const getFigure = (figure) => {
  const Figures = {
    "Aida": "Открывающая позиция",
    "Alemana": "Поворотная фигура",
    "Fan": "Открытие в веерную позицию",
    "FootChange": "Базовая смена стопы",
    "HandToHandL": "Рука к руке влево",
    "HandToHandR": "Рука к руке вправо",
    "HockyStick": "Хоккейная клюшка",
    "NaturalTop": "Натуральный топ",
    "NewYorkL": "Нью-Йорк влево",
    "NewYorkR": "Нью-Йорк вправо",
    "NotPerforming": "Нейтральная/отдыхающая позиция",
    "OpenBasic": "Открытый базовый",
    "OpeningOut": "Открытие наружу",
    "SpotTurn": "Спот поворот"
  };
  return Figures[figure];
}


const formatTime = (time) => {
  const hours = Math.floor(time / 3600)
  const minutes = Math.floor((time % 3600) / 60)
  const seconds = Math.floor(time % 60)
  const milliseconds = Math.floor((time % 1) * 1000)

  const hStr = String(hours).padStart(2, '0')
  const mStr = String(minutes).padStart(2, '0')
  const sStr = String(seconds).padStart(2, '0')
  const msStr = String(milliseconds).padStart(3, '0')

  return `${hStr}:${mStr}:${sStr}.${msStr}`
}

const onTimeUpdate = () => {
  if (videoPlayer.value) {
    timeDisplay.value = formatTime(videoPlayer.value.currentTime)
  }
}



const triggerUpload = () => {
  fileInput.value.click()
}

const handleVideoUpload = (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  if (!file.type.startsWith('video/')) {
    alert('Пожалуйста, выберите видеофайл.')
    return
  }

  uploadedFile.value = file;
  const url = URL.createObjectURL(file);
  const tmpVideo = document.createElement('video');

  tmpVideo.preload = 'metadata';
  tmpVideo.src = url;
  tmpVideo.onloadedmetadata = () => {
    // Проверяем нужные параметры видео
    if (tmpVideo.videoWidth !== 1920 || tmpVideo.videoHeight !== 1080) {
      alert(`Видео не соответствует требуемому разрешению 1920x1080. Загружено: ${tmpVideo.videoWidth}x${tmpVideo.videoHeight}`);
      URL.revokeObjectURL(url);
      return;
    }
    uploadedVideo.value = url;
    processedVideoUrl.value = null
    message.value = ''
    analysisResult.value = null;

     // Установим endTime равным длительности видео с миллисекундами
  const duration = tmpVideo.duration; // длительность в секундах с дробной частью
  const hours = Math.floor(duration / 3600);
  const minutes = Math.floor((duration % 3600) / 60);
  const seconds = Math.floor(duration % 60);
  const milliseconds = Math.floor((duration % 1) * 1000);
  endTime.value = 
    String(hours).padStart(2, '0') + ':' +
    String(minutes).padStart(2, '0') + ':' +
    String(seconds).padStart(2, '0') + '.' +
    String(milliseconds).padStart(3, '0');
  };
  tmpVideo.onerror = () => {
    alert('Ошибка при загрузке видео.');
    URL.revokeObjectURL(url);
  };
};

const handleDrop = (e) => {
  const file = e.dataTransfer?.files[0];
  if (!file) return;

  // Проверка на видеофайл
  if (!file.type.startsWith('video/')) {
    alert('Пожалуйста, выберите видеофайл.');
    return;
  }
  console.log("File", file)
  uploadedFile.value = file;
  // Проверка разрешения видео и последующая установка
  const url = URL.createObjectURL(file);
  const tmpVideo = document.createElement('video');
  tmpVideo.preload = 'metadata';
  tmpVideo.src = url;
  tmpVideo.onloadedmetadata = () => {
    if (tmpVideo.videoWidth !== 1920 || tmpVideo.videoHeight !== 1080) {
      alert(`Видео не соответствует требуемому разрешению 1920x1080. Загружено: ${tmpVideo.videoWidth}x${tmpVideo.videoHeight}`);
      URL.revokeObjectURL(url);
      return;
    }
    uploadedVideo.value = url;
    processedVideoUrl.value = null;
    message.value = '';
    analysisResult.value = null;

    // Установим endTime равным длительности видео с миллисекундами
    const duration = tmpVideo.duration; // длительность в секундах с дробной частью
    const hours = Math.floor(duration / 3600);
    const minutes = Math.floor((duration % 3600) / 60);
    const seconds = Math.floor(duration % 60);
    const milliseconds = Math.floor((duration % 1) * 1000);
    endTime.value = 
      String(hours).padStart(2, '0') + ':' +
      String(minutes).padStart(2, '0') + ':' +
      String(seconds).padStart(2, '0') + '.' +
      String(milliseconds).padStart(3, '0');
  };
  tmpVideo.onerror = () => {
    alert('Ошибка при загрузке видео.');
    URL.revokeObjectURL(url);
  };
};


import { FFmpeg } from '/node_modules/@ffmpeg/ffmpeg/dist/esm/index.js';
import { fetchFile } from '/node_modules/@ffmpeg/util/dist/esm/index.js';

const trimVideo = async (blob, startTime, endTime) => {
  const ffmpeg = new FFmpeg();
  await ffmpeg.load();
  await ffmpeg.writeFile('input.mp4', await fetchFile(blob));
  await ffmpeg.exec([
    '-ss', startTime.toFixed(3),
    '-i', 'input.mp4',
    '-to', endTime.toFixed(3),
    '-c:v', 'copy',
    '-c:a', 'copy',
    '-avoid_negative_ts', 'make_zero',
    '-copyts',
    'output_lossless.mp4'
  ]);
  const data = await ffmpeg.readFile('output_lossless.mp4');
  return new Blob([data.buffer], { type: 'video/mp4' });
};

const trimmedVideoUrl = ref(null); // новое состояние для обрезанного видео
const trimmedBlob = ref(null);
const cutVideo = async () => {
  if (!uploadedFile.value) {
    alert('Видео не загружено!');
    return;
  }

  const parseTimeToSeconds = (timeStr) => {
    const [h, m, sMs] = timeStr.split(':');
    const [s, ms] = sMs.split('.');
    return Number(h) * 3600 + Number(m) * 60 + Number(s) + Number(ms) / 1000;
  };

  const startSec = parseTimeToSeconds(startTime.value);
  const endSec = parseTimeToSeconds(endTime.value);

  if (endSec <= startSec) {
    alert('Время конца должно быть больше начала');
    return;
  }

  try {
    isAnalyzing.value = true;

    const file = uploadedFile.value;
    trimmedBlob.value = await trimVideo(file, startSec, endSec);
    trimmedVideoUrl.value = URL.createObjectURL(trimmedBlob.value);
    await handleAnalyze(trimmedVideoUrl.value.split('/').pop());
  } catch (err) {
    alert('Ошибка при обрезке видео: ' + err.message);
  } finally {
    isAnalyzing.value = false;
  }
  
};

const handleAnalyze = async (blobValue) => {
  
  if (!uploadedFile.value) {
    alert('Видео не загружено!');
    return;
  }
  isAnalyzing.value = true;

  try {
    const formData=new FormData();
    const filename = `${blobValue}.mp4`
    formData.append('video', trimmedBlob.value, filename);
    // Отправляем POST-запрос на сервер
    await axios.post(`http://localhost:3000/video/${filename}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    socket.emit('register-upload', `http://localhost:3000/uploads/${filename}`)
    message.value = 'Видео успешно отправлено на анализ.';
  } catch (error) {
    alert('Ошибка при отправке видео: ' + error.message);
  } finally {
    isAnalyzing.value = false;
  }
};


</script>
