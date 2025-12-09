<template>
  <div class="time-picker-ms">
    <div class="time-inputs flex items-center gap-2">
      <input
        v-model.number="hours"
        type="number"
        min="0"
        max="23"
        placeholder="00"
        class="w-16 bg-gray-700 border border-gray-600 rounded-lg px-3 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 text-center"
        @input="updateTime"
      />
      <span class="text-gray-400 font-bold">:</span>
      <input
        v-model.number="minutes"
        type="number"
        min="0"
        max="59"
        placeholder="00"
        class="w-16 bg-gray-700 border border-gray-600 rounded-lg px-3 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 text-center"
        @input="updateTime"
      />
      <span class="text-gray-400 font-bold">:</span>
      <input
        v-model.number="seconds"
        type="number"
        min="0"
        max="59"
        placeholder="00"
        class="w-16 bg-gray-700 border border-gray-600 rounded-lg px-3 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 text-center"
        @input="updateTime"
      />
      <span class="text-gray-400 font-bold">.</span>
      <input
        v-model.number="milliseconds"
        type="number"
        min="0"
        max="999"
        placeholder="000"
        class="w-20 bg-gray-700 border border-gray-600 rounded-lg px-3 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-pink-500 text-center"
        @input="updateTime"
      />
    </div>
    <p class="text-sm text-gray-500 mt-2">Формат: часы:минуты:секунды.миллисекунды</p>
  </div>
</template>

<script setup>
import { ref, watch, computed } from 'vue'

const props = defineProps({
  modelValue: {
    type: String,
    default: '00:00:00.000'
  }
})

const emit = defineEmits(['update:modelValue'])

const hours = ref(0)
const minutes = ref(0)
const seconds = ref(0)
const milliseconds = ref(0)

const formattedTime = computed(() => {
  const h = String(hours.value || 0).padStart(2, '0')
  const m = String(minutes.value || 0).padStart(2, '0')
  const s = String(seconds.value || 0).padStart(2, '0')
  const ms = String(milliseconds.value || 0).padStart(3, '0')
  return `${h}:${m}:${s}.${ms}`
})

const parseTime = (timeString) => {
  const match = timeString.match(/(\d{2}):(\d{2}):(\d{2})\.(\d{3})/)
  if (match) {
    hours.value = parseInt(match[1])
    minutes.value = parseInt(match[2])
    seconds.value = parseInt(match[3])
    milliseconds.value = parseInt(match[4])
  }
}

const updateTime = () => {
  hours.value = Math.max(0, Math.min(23, hours.value || 0))
  minutes.value = Math.max(0, Math.min(59, minutes.value || 0))
  seconds.value = Math.max(0, Math.min(59, seconds.value || 0))
  milliseconds.value = Math.max(0, Math.min(999, milliseconds.value || 0))
  
  emit('update:modelValue', formattedTime.value)
}

watch(() => props.modelValue, (newValue) => {
  parseTime(newValue)
}, { immediate: true })
</script>