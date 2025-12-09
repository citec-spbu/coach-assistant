import axios from 'axios'
import fs from 'fs'
import FormData from 'form-data'

const KOA_URL = 'http://localhost:3000'

async function uploadBadFile() {
  const form = new FormData()
  // создаёшь маленький текстовый файл tests/not_video.txt
  form.append('video', fs.createReadStream('tests/node/not_video.txt'))

  const url = `${KOA_URL}/video/not_video.txt`
  const res = await axios.post(url, form, {
    validateStatus: () => true,      // не бросать по 4xx
    headers: form.getHeaders()
  })

  if (res.status !== 400) {
    throw new Error(`ожидали 400, а получили ${res.status}`)
  }
  if (!res.data?.error) {
    throw new Error('ожидали error в ответе')
  }

  console.log('OK: не‑видео даёт 400 и сообщение об ошибке')
}

async function main() {
  // позитивный сценарий (то, что уже работает)
  const form = new FormData()
  form.append('video', fs.createReadStream('tests/node/test.mp4'))
  const url = `${KOA_URL}/video/test.mp4`
  const res = await axios.post(url, form, { headers: form.getHeaders() })
  if (res.status !== 200 || !res.data.success) {
    throw new Error('Upload failed')
  }
  const uploadUrl = `http://localhost:3000/uploads/test.mp4`
  console.log('uploadUrl =', uploadUrl)

  // ожидание done, как у тебя
  const start = Date.now()
  while (Date.now() - start < 60000) {
    const statusRes = await axios.get(`${KOA_URL}/api/get`, {
      params: { upload_url: uploadUrl }
    })
    if (statusRes.status === 200 && statusRes.data.data?.status === 'done') {
      console.log('info =', statusRes.data.data)
      break
    }
    await new Promise(r => setTimeout(r, 2000))
  }
  console.log('OK: API цепочка отработала')

  // новый негативный сценарий
  await uploadBadFile()
}

main()
  .then(() => {
    console.log('ALL TESTS PASSED')
    process.exit(0)
  })
  .catch(err => {
    console.error('TEST FAILED:', err)
    process.exit(1)
  })
