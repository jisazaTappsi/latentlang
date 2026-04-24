<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import axios from 'axios'

type InterpretResponse = {
  result: string
  error: string | null
  symbols?: Record<string, unknown> | null
}

type HistoryEntry = {
  type: 'input' | 'output' | 'error'
  text: string
  symbol_table?: string | null
}

const history = ref<HistoryEntry[]>([
  { type: 'output', text: 'LatentLang runs a Transformer interpreter: type code with arbitrary syntax it will just figure it out :).'}, {type: 'output', text: 'For example, type "3 times 2" or "3.mul 2" both cases will work, even though the syntax is not precisely defined' },
  { type: 'output', text: 'Type "clear" to reset.' },
])

let symbols: Record<string, unknown> | null = null
const input = ref('')
const loading = ref(false)
const inputEl = ref<HTMLInputElement | null>(null)
const terminalEl = ref<HTMLDivElement | null>(null)
const commandHistory = ref<string[]>([])
const commandHistoryIndex = ref(-1)
const draftInput = ref('')

const config = useRuntimeConfig()

function interpretUrl() {
  const base = String(config.public.apiBase).replace(/\/$/, '')
  return `${base}/interpret`
}

onMounted(() => inputEl.value?.focus())

async function scrollToBottom() {
  await nextTick()
  if (terminalEl.value) {
    terminalEl.value.scrollTop = terminalEl.value.scrollHeight
  }
}

async function submit() {
  const code = input.value.trim()
  if (!code) return
  commandHistory.value.push(code)
  commandHistoryIndex.value = -1
  draftInput.value = ''

  if (code === 'clear') {
    history.value = []
    input.value = ''
    await nextTick()
    focusInput()
    return
  }

  history.value.push({ type: 'input', text: code })
  input.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const { data } = await axios.post<InterpretResponse>(interpretUrl(), { code, symbols })

    if (data.error) {
      history.value.push({ type: 'error', text: data.error })
    } else {
      history.value.push({ type: 'output', text: data.result })
    }
    symbols = data.symbols ?? null

  } catch (err: unknown) {
    const message =
      axios.isAxiosError(err) && err.response?.data
        ? typeof err.response.data === 'string'
          ? err.response.data
          : JSON.stringify(err.response.data)
        : err instanceof Error
          ? err.message
          : 'Request failed'
    history.value.push({ type: 'error', text: message })
  } finally {
    loading.value = false
    await scrollToBottom()
    await nextTick()
    focusInput()
  }
}

function focusInput() {
  inputEl.value?.focus()
}

function onTerminalClick() {
  const selection = window.getSelection()
  if (selection && !selection.isCollapsed) return
  focusInput()
}

function moveCaretToEnd() {
  nextTick(() => {
    const el = inputEl.value
    if (!el) return
    const len = el.value.length
    el.setSelectionRange(len, len)
  })
}

function onHistoryUp() {
  if (loading.value || commandHistory.value.length === 0) return

  if (commandHistoryIndex.value === -1) {
    draftInput.value = input.value
    commandHistoryIndex.value = commandHistory.value.length - 1
  } else if (commandHistoryIndex.value > 0) {
    commandHistoryIndex.value -= 1
  }

  input.value = commandHistory.value[commandHistoryIndex.value] ?? ''
  moveCaretToEnd()
}

function onHistoryDown() {
  if (loading.value || commandHistory.value.length === 0 || commandHistoryIndex.value === -1) return

  if (commandHistoryIndex.value < commandHistory.value.length - 1) {
    commandHistoryIndex.value += 1
    input.value = commandHistory.value[commandHistoryIndex.value] ?? ''
  } else {
    commandHistoryIndex.value = -1
    input.value = draftInput.value
  }

  moveCaretToEnd()
}
</script>

<template>
  <div class="terminal" @click="onTerminalClick" ref="terminalEl">
    <div
      v-for="(entry, i) in history"
      :key="i"
      :class="['line', entry.type]"
    >
      <span v-if="entry.type === 'input'" class="prompt">&gt;&nbsp;</span>
      <span>{{ entry.text }}</span>
    </div>

    <div class="line input-line">
      <span class="prompt">&gt;&nbsp;</span>
      <input
        ref="inputEl"
        v-model="input"
        :disabled="loading"
        @keydown.enter="submit"
        @keydown.up.prevent="onHistoryUp"
        @keydown.down.prevent="onHistoryDown"
        autocomplete="off"
        spellcheck="false"
        class="input-field"
      />
    </div>
  </div>
</template>

<style scoped>
.terminal {
  background: #0d0d0d;
  color: #d4d4d4;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
  line-height: 1.6;
  padding: 16px;
  height: 100%;
  overflow-y: auto;
  cursor: text;
  box-sizing: border-box;
}

.line {
  display: flex;
  align-items: flex-start;
  white-space: pre-wrap;
  word-break: break-all;
  margin-bottom: 2px;
}

.line.input .prompt,
.input-line .prompt {
  color: #4ec9b0;
  flex-shrink: 0;
}

.line.output {
  color: #d4d4d4;
}

.line.error {
  color: #f44747;
}

.input-line {
  display: flex;
  align-items: center;
  margin-top: 4px;
}

.input-field {
  background: transparent;
  border: none;
  outline: none;
  color: #d4d4d4;
  font-family: inherit;
  font-size: inherit;
  flex: 1;
  caret-color: #d4d4d4;
}

.input-field:disabled {
  opacity: 0.5;
}
</style>
