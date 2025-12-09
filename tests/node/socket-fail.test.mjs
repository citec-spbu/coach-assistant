console.log('socket-fail.test.mjs START')

import { io } from 'socket.io-client'
import axios from 'axios'

const KOA_HTTP = 'http://localhost:3000'
const KOA_WS = 'http://localhost:3000'

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
    socket.on('connect_error', err => {
      console.error('connect_error', err)
      reject(err)
    })
  })

  const uploadUrl = 'http://localhost:3000/uploads/socket_fail.mp4'
  console.log('register uploadUrl =', uploadUrl)

  socket.emit('register-upload', uploadUrl)

  // Имитируем "плохой" результат обработки
  await axios.post(`${KOA_HTTP}/api/result`, {
    status: 'failed',
    upload_url: uploadUrl,
    download_url: null,
    metadata: null
  })

  const payload = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error('timeout waiting for video-ready (failed)')),
      10000
    )

    socket.on('video-ready', data => {
      console.log('video-ready (failed) =', data)
      clearTimeout(timeout)
      resolve(data)
    })
  })

  if (!payload || payload.status !== 'failed') {
    throw new Error('ожидали status="failed", а получили: ' + JSON.stringify(payload))
  }

  console.log('OK: негативный сценарий video-ready (failed) прошёл проверку')
  socket.disconnect()
}

main()
  .then(() => {
    console.log('socket-fail.test.mjs END')
    process.exit(0)
  })
  .catch(err => {
    console.error('TEST FAILED:', err)
    process.exit(1)
  })
