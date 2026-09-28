import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // docker(Windows 폴더 마운트)에서는 파일 변경 알림이 안 와서 고친 코드가 반영되지 않는다.
    // docker-compose.yml 이 VITE_USE_POLLING=true 를 넘기면 주기적으로 파일을 확인한다.
    watch: process.env.VITE_USE_POLLING === 'true' ? { usePolling: true, interval: 300 } : undefined,
  },
})
