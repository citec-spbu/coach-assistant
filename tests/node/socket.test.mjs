// tests/socket.test.mjs
import { io } from 'socket.io-client'
import axios from 'axios'
import fs from 'fs'
import FormData from 'form-data'
console.log('socket.test.mjs START')

const KOA_HTTP = 'http://localhost:3000'
const KOA_WS = 'http://localhost:3000'

async function uploadVideo(uploadName) {
  const form = new FormData()
  form.append('video', fs.createReadStream('tests/node/test.mp4'))

  const url = `${KOA_HTTP}/video/${uploadName}`
  const res = await axios.post(url, form, {
    headers: form.getHeaders()
  })

  if (res.status !== 200 || !res.data.success) {
    throw new Error('upload failed: ' + res.status)
  }

  return `http://localhost:3000/uploads/${uploadName}`
}

async function main() {
  const socket = io(KOA_WS, {
    transports: ['websocket'],
    reconnection: false
  })

  await new Promise((resolve, reject) => {
    socket.on('connect', () => {
      console.log('socket connected', socket.id)
      resolve()
    })
    socket.on('connect_error', reject)
  })

  const uploadName = 'socket_test.mp4'
  const uploadUrl = await uploadVideo(uploadName)
  console.log('uploadUrl =', uploadUrl)

  // регистрируемся на сервере, чтобы он знал, куда слать video-ready
  socket.emit('register-upload', uploadUrl)

  const result = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('timeout waiting for video-ready')),
      60000
    )

    socket.on('video-ready', payload => {
      clearTimeout(timeout)
      resolve(payload)
    })

    socket.on('error', err => {
      clearTimeout(timeout)
      reject(err)
    })
  })

  console.log('video-ready =', result)

  if (!result || result.status !== 'done') {
    throw new Error('expected status=done, got ' + JSON.stringify(result))
  }
  if (!result.upload_url || !result.download_url) {
    throw new Error('missing upload_url/download_url in payload')
  }

  socket.disconnect()
  console.log('OK: video-ready событие прошло проверку')
}

main()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('TEST FAILED:', err)
    process.exit(1)
  })
console.log('socket.test.mjs END')
