<template>
  <div class="page">
    <header class="header">
      <span class="title">LatentLang</span>
    </header>
    <main class="main">
      <Repl />
    </main>
  </div>
</template>

<style scoped>
.page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #0d0d0d;
}

.header {
  display: flex;
  align-items: center;
  padding: 8px 16px;
  background: #1e1e1e;
  border-bottom: 1px solid #333;
  flex-shrink: 0;
}

.title {
  color: #4ec9b0;
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  letter-spacing: 2px;
}

.main {
  flex: 1;
  overflow: hidden;
}
</style>
